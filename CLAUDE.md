# M4 Pro Inference Project

SGLang with native MLX backend on Apple M4 Pro (64GB unified memory).

**All inference MUST use SGLang with the MLX backend.** Set `SGLANG_USE_MLX=1` for all operations.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, known issues |
| [rules-for-agents.md](rules-for-agents.md) | Apple Silicon constraints, launch rules, MLX specifics |

## Key Commands
```bash
scripts/setup.sh                       # Full setup (venv, SGLang from main, MLX deps)
scripts/launch.sh devstral             # Devstral 24B 4-bit
scripts/launch.sh coder-30b            # Coder-30B MoE 4-bit
scripts/launch.sh coder-next           # Coder-Next 80B 4-bit
scripts/launch.sh gemma4               # Gemma 4 26B MoE 4-bit
scripts/launch.sh qwen35               # Qwen3.5-27B 4-bit
```

## Critical Rules
- **SGLang + MLX only** — all models must run on SGLang with `SGLANG_USE_MLX=1`
- **No tensor parallelism** — MLX runs on a single unified memory device
- **Greedy sampling only** — MLX backend uses argmax; temperature/top-p not yet supported
- **MLX-format models required** — AWQ/GPTQ models from other platforms won't work; use `mlx_lm.convert` or download from `mlx-community/` on HuggingFace
- Always source `scripts/common.sh` before launching
- **Model status and benchmarks** are in README.md (single source of truth)
