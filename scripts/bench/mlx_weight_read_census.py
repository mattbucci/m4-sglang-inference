#!/usr/bin/env python3
"""Bytes of weights one decode step reads, by family, from the safetensors
headers of an MLX checkpoint (no model load, no GPU).

A decode step of one token streams every weight its matmuls touch: the whole
of each dense projection and the LM head, and for a MoE layer only the routed
experts (num_experts_per_tok of num_experts). The embedding contributes one
row and the vision tower nothing. Divided by the rate the GPU sustains on
quantized matmuls (mlx_launch_microbench.py reports it), the total is the
time a step spends reading weights, which is the floor that launch folding
and scheduling cannot go under.

usage: mlx_weight_read_census.py <model dir> [out json | out jsonl] [GB/s csv]
  e.g. mlx_weight_read_census.py ~/.cache/.../Qwen3.6-35B-A3B-4bit/snapshots/<rev> census.json 178,239
  A .jsonl output is appended to, one row per checkpoint.
"""
import collections, glob, json, os, struct, sys

BYTES = {"U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
         "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8, "BOOL": 1}


def family(name: str) -> str:
    if "vision" in name or "visual" in name:
        return "vision tower"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "switch_mlp" in name or ".experts." in name:
        return "routed experts"
    if "shared_expert" in name:
        return "shared expert"
    if ".mlp.gate." in name:
        return "router gate"
    if "linear_attn" in name or "mixer" in name:
        return "recurrent mixer"
    if "self_attn" in name or ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "dense mlp"
    return "other"


def main() -> None:
    root = os.path.expanduser(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    rates = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "178,239").split(",")]
    cfg = json.load(open(os.path.join(root, "config.json")))
    text = cfg.get("text_config", cfg)
    n_exp = int(text.get("num_experts", 0) or text.get("n_routed_experts", 0) or 0)
    top_k = int(text.get("num_experts_per_tok", 0) or 0)
    tied = bool(cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", False)))

    disk = collections.Counter()
    for path in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with open(path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            size = BYTES[meta["dtype"]]
            for dim in meta["shape"]:
                size *= dim
            disk[family(name)] += size

    read = {}
    for fam, size in disk.items():
        if fam == "vision tower":
            read[fam] = 0
        elif fam == "embedding":
            # one row per token; a tied head reads the table once as the LM head
            read[fam] = size if tied and "lm_head" not in disk else 0
        elif fam == "routed experts" and n_exp:
            read[fam] = size * top_k // n_exp
        else:
            read[fam] = size
    total = sum(read.values())

    rows = [{"family": fam, "disk_mb": round(disk[fam] / 1e6, 1),
             "read_mb_per_token": round(read[fam] / 1e6, 1),
             "share": round(read[fam] / total, 4)}
            for fam in sorted(disk, key=lambda f: -read[f])]
    parts = root.rstrip("/").split("/")
    hub = [p for p in parts if p.startswith("models--")]
    name = hub[-1][len("models--"):].replace("--", "/") if hub else parts[-1]
    result = {"model": name,
              "num_experts": n_exp, "num_experts_per_tok": top_k,
              "disk_mb": round(sum(disk.values()) / 1e6, 1),
              "read_mb_per_token": round(total / 1e6, 1),
              "read_ms_at_gb_per_s": {str(int(r)): round(total / 1e9 / r * 1e3, 2) for r in rates},
              "families": rows}

    print(f"{'family':18s} {'on disk MB':>11s} {'read MB/token':>14s} {'share':>7s}")
    for row in rows:
        print(f"{row['family']:18s} {row['disk_mb']:11.1f} {row['read_mb_per_token']:14.1f} {row['share']*100:6.1f}%")
    print(f"{'total':18s} {result['disk_mb']:11.1f} {result['read_mb_per_token']:14.1f}")
    for rate, ms in result["read_ms_at_gb_per_s"].items():
        print(f"at {rate} GB/s: {ms} ms per token")
    if out and out.endswith(".jsonl"):
        with open(out, "a") as fh:
            fh.write(json.dumps(result) + "\n")
    elif out:
        json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
