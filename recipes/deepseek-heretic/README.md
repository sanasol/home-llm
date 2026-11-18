# Heretic attention-bank recipe for DeepSeek Sushi

This is the **38-bank attention overlay** used to build
[sanasol2008/DeepSeek-V4.1-Flash-1017-Heretic-Sushi-STQ-1.75bpw](https://huggingface.co/sanasol2008/DeepSeek-V4.1-Flash-1017-Heretic-Sushi-STQ-1.75bpw),
not a full quantization of the Heretic checkpoint. Official routed experts,
all original F16 scales, headers, and every non-target byte are retained.
Only `layers.2..39.attn.wo_b.trellis` is re-encoded, at the original SPEC1-w15
K4 rate (`n64`), using edited native FP8 attention weights and base E8M0 scales.

Encoding is F32 uncalibrated SPEC1 least-squares, not Sushi's private Sashimi
encoder or its calibrated recipe. Do not use these scripts for other donors.
No reconstructed edit direction is substituted for actual Heretic tensors.

## Pinned inputs

| Input | Immutable revision |
|---|---|
| [Official Sushi STQ 1.75bpw root](https://huggingface.co/beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw) | `89630812557263bb7a7582fb1f30455573511155` |
| [pqhaz native FP8 Heretic patch](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic), `patch/heretic_patch.safetensors` | `12d58ad0878b5008e80d720251adb969a0069412` |
| [Native base FP8 scales](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `08c8657f02291e11b6f6b63e0e16014f8e748518` |
| [Public Sushi tables/decoder](https://github.com/beamivalice/sushi/tree/d5e876edcb1538a347041b1b7cc28a0e6e8f0af6) | `d5e876edcb1538a347041b1b7cc28a0e6e8f0af6` |

All range calls in this recipe use explicit revisions. The public recipe removes
unrelated historical donor paths and pins formerly implicit native-header reads;
it is a cleaned reproducibility recipe, not a claim that its source is byte-identical
to the archived producer scripts. Floating-point results may depend on Torch,
CUDA and device versions. Verify a new build independently.

## Environment and CPU checks

Use Python 3.11+ and a fresh environment. Install a CUDA-compatible Torch wheel
from [PyTorch](https://pytorch.org/get-started/locally/) for full encoding; no GPU
is needed for the small synthetic checks. No GGUF package is needed.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy torch
cd recipes/deepseek-heretic
python -m unittest discover -p 'test_recipe.py' -v
python build.py --help
python assemble_root.py --help
export DS_WORK="$PWD/heretic-work"
```

Authentication uses `HF_TOKEN` or `~/.cache/huggingface/token`. Obtain permission
for gated inputs yourself; never put credentials into scripts, URLs or commits.
HTTP access uses an HTTPS-only opener: credentials attach only to the initial
`huggingface.co` origin, cross-origin redirects strip Authorization, and HTTPS-to-HTTP
downgrades are rejected. Signed CDN redirects remain supported. No inference endpoint is used. Model downloads and GPU compute incur their normal
network/hardware costs; these scripts do not rent machines or upload anything.

## 1. Download the official root once

The full official root is about 310 GiB. Keep room for approximately 4 GiB of
replacement trunks plus temporary files and an 8 GiB reserve. Write directly into
one root, avoiding a duplicate HF cache:

```sh
python assemble_root.py download "$DS_WORK/official"
```

The downloader checks LFS SHA256 or the Git blob identity at the pinned revision.
It writes `OFFICIAL_VERIFIED.json`. Large files stream in bounded chunks with two
workers; a failed partial file restarts on rerun, while verified large files are
reused. Downloads are explicit, not part of `--help` or CPU tests.

The encoder otherwise fetches native tensors/header ranges and official trunk
banks on demand. Base FP8 codes and E8M0 scales use 32x32 blocks. The Heretic patch
must contain edited F8_E4M3 codes without separate scale tensors; unexpected scale
or geometry changes fail closed.

## 2. One CPU slab, then full GPU encoding

The slab uses the first 128 input rows of one target as a bounded algorithm smoke.
It is still a real donor fetch and may be slow on CPU; it is not model inference:

```sh
python build.py slab heretic 2 wo_b
```

One CUDA GPU, conservative starting batch:

```sh
CUDA_VISIBLE_DEVICES=0 python build.py encode heretic --part 0/1 --batch 256
```

For eight available GPUs, dispatch disjoint layer partitions (no rentals/startup
or production-process control is hidden here):

```sh
set -euo pipefail
pids=""
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$i" python build.py encode heretic --part "$i/8" --batch 256 \
    > "$DS_WORK/encode-$i.log" 2>&1 &
  pids="$pids $!"
done
failed=0
for pid in $pids; do wait "$pid" || failed=1; done
test "$failed" = 0
```

Run each partition only once at a time. Resume the same work directory and partition
commands to reuse completed `.npy` outputs; do not change seeds/code/dependencies
or mix outputs from another run. Use a fresh `DS_WORK` for changed settings.
The archived full production duration is not established here; no ETA is invented.
Per-bank metrics report actual encoding seconds and numerical proxy errors.
These are not full-model quality scores, refusal rates, or lossless fidelity proof.

## 3. Assemble and verify the 38 replacement trunks

```sh
python build.py assemble heretic
python assemble_root.py assemble "$DS_WORK/official" "$DS_WORK/pack/heretic" "$DS_WORK/heretic"
```

Assembly requires exactly 38 trunks and verifies that only the intended trellis
byte range changed. Original scales, headers and all non-target bytes must match.
The full official root is rehashed before assembly. The final root copies modified
trunks and hardlinks only unchanged official files on the same filesystem. It never
patches those hardlinks in place. Keep the source/output roots disjoint and use a
fresh output name. If your filesystem cannot hardlink, use separate verified copies
instead and budget the full additional disk space; this helper does not silently
fall back to duplicating 310 GiB.

The copied official config/index describes the unchanged public tensor layout;
`HERETIC_BUILD_PROVENANCE.json` records the build and each final file's SHA256.
The official `pack_revision`/quant stamps are preserved because tensor names,
shapes, rates, and headers remain the official overlay layout; provenance explicitly
identifies the 38 replacements. This matches the overlay construction, not a new
whole-model revision-stamping protocol. Do not merge unrelated expert shards.

**Do not run an in-place repacker against a hardlinked output:** any unchanged
shared shard could mutate the official source. The published Heretic layout loaded
without a custom repack in its recorded smoke. If a future engine needs rewriting,
first make independent copies or filesystem reflinks and preserve the download hashes.

## Runtime and scope

Use the linked model card for the tested isolated Sushi 1.6.0 Apple Silicon setup,
28 GiB SSD budget, ctx2048, `--no-mtp --no-vision`, loopback binding and bounded
thinking-off smoke commands. Runtime smoke is not general quality validation.
This recipe does not execute prompts, evaluate harmful outputs, start engines,
change machine settings or publish model weights.

## Attribution

Recipe code is MIT (see `LICENSE`). Public Sushi codebook/decoder conventions are
attributed in `NOTICE.md` and `LICENSE-SUSHI`. Model weights retain their upstream
licenses; keep the pinned official/donor licenses in redistributed model roots.
No GPL implementation is included and no private Sashimi recipe is claimed.
