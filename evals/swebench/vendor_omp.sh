#!/usr/bin/env bash
# Install the pinned oh-my-pi (omp) binary the SWE-bench harness drives:
# evals/swebench/bin/omp (gitignored, ~196 MB).
#
#   bash evals/swebench/vendor_omp.sh            # GitHub release asset
#   OMP_VERSION=18.12.2 bash evals/swebench/vendor_omp.sh
#
# omp is a bun-compiled pi-family coding agent shipped as one standalone
# binary per platform. The version is pinned to the 3090 rig's scaffold pin
# so the two rigs' omp cells stay comparable; the sha256 comes from the
# release's SHA256SUMS.txt. The npm package needs bun and the project's
# installer pulls the same prebuilt binary, so the asset is fetched directly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VER="${OMP_VERSION:-18.12.2}"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) ASSET=omp-darwin-arm64 ;;
  Darwin-x86_64) ASSET=omp-darwin-x64 ;;
  Linux-x86_64) ASSET=omp-linux-x64 ;;
  Linux-aarch64) ASSET=omp-linux-arm64 ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
# sha256 per release and asset, from SHA256SUMS.txt (macOS bash 3.2: no associative arrays)
case "$VER/$ASSET" in
  18.12.2/omp-darwin-arm64) WANT=abd9a5b0fef77f4204078e348e233faa0dd3d5d81719cf2b6a9b9478c10eee48 ;;
  18.12.2/omp-darwin-x64)   WANT=dcd3fa827ae73bdb471bb838171ef1c9c5be665a68eaec45ffc6d6e5cf128b92 ;;
  18.12.2/omp-linux-x64)    WANT=9513ef9f19d7991aac6ae0cb6077bcf16662092ddde861c002a91cedc6f47f9a ;;
  18.12.2/omp-linux-arm64)  WANT=967d64fce514586e94c9ac5f978743b4fce29b999c8763747ce4754811407d1a ;;
  *) echo "no pinned sha256 for omp $VER $ASSET — add it from the release's SHA256SUMS.txt first" >&2; exit 1 ;;
esac
OUT="$HERE/bin/omp"
mkdir -p "$HERE/bin"
sum() { shasum -a 256 "$1" | cut -d' ' -f1; }
if [[ -x "$OUT" && "$(sum "$OUT")" == "$WANT" ]]; then
  echo "$OUT  omp/$VER  sha256=$WANT (already present)"; exit 0
fi
URL="https://github.com/can1357/oh-my-pi/releases/download/v${VER}/${ASSET}"
curl -fL --retry 3 -o "$OUT.partial" "$URL"
got="$(sum "$OUT.partial")"
if [[ "$got" != "$WANT" ]]; then
  echo "sha256 mismatch for $ASSET $VER: got $got want $WANT" >&2; rm -f "$OUT.partial"; exit 1
fi
chmod 0755 "$OUT.partial" && mv "$OUT.partial" "$OUT"
test "$("$OUT" --version 2>&1)" = "omp/${VER}"
echo "$OUT  omp/$VER  sha256=$got"
