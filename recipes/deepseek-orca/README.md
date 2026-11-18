# Orca GGUF to Sushi: released hybrid CPU/CUDA recipe

This is the public, architecture-specific recipe behind the original
[Orca Sushi model](https://huggingface.co/sanasol2008/DeepSeek-V4.1-Flash-1017-Orca-Sushi-STQ-1.75bpw).
Released weight commit: `3f2e0985a9b9380ffc83e83485dcbda6874b959c`.
It is **not** the obsolete rank-one/trunk overlay and does not reuse official
Sushi expert weights. All donor expert projections are read and encoded afresh.

The released recipe uses CPU NumPy F64 calibration/feedback and CUDA F32 trellis
search. "Hybrid CPU/CUDA" describes encoding, not a mixture of model checkpoints.
The later fully GPU-resident experiment was cancelled before a complete published
candidate; that experimental implementation is deliberately not included here.
This is not Sashimi, not Sushi's private conversion recipe, and not a Qwen recipe.

## Inputs and fidelity

- Donor: `orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF`, pinned
  `83144a667924660c6ba0095585fd039d8daa9b91`, `Mix-256G/` plus public `imatrix.gguf`.
- Native config/tokenizer assets: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, pinned
  `08c8657f02291e11b6f6b63e0e16014f8e748518`. Exact donor token IDs/merges/map and
  architecture were checked before reuse; no base-model weights are grafted.
- Public codebook/consumer reference: Sushi
  `d5e876edcb1538a347041b1b7cc28a0e6e8f0af6`.

The donor is already quantized GGUF, not BF16. Re-encoding adds loss. Routed
PAIR1-w16-v1 n28 is 1.75 payload bits/weight, **not whole-pack average bpw**.
337 calibrated SPEC1-w15 n64/K4 trunk banks, 464 dense F32 native trunk tensors,
F32 embedding/head, F16 scales, BF16 vision matrices and original F32 vision
vectors, and two FP8 lookup tables contribute additional bytes. DSpark is absent
from the donor and is omitted, never borrowed from another checkpoint.

Engram IQ4_NL rows are dequantized then requantized to native E4M3FN/E8M0: lossy,
not recovery of original FP8. Fused donor q/k is represented explicitly as F32
product times F32 ones for the pinned consumer; original factors are not inferred.

Four unvisited experts use **explicit uncalibrated identity importance at n28**:
L21/E207, L24/E350, L25/E85, L27/E260. All other routed projections use native-input
diagonals, normalized by common layer count614400 (not per-expert normalization,
which would erase routing frequency). Counts/inputs are generated from the pinned
public imatrix; no private NPZ is required.

## Environment and resource budget

Linux, Python3.11+, NumPy, CUDA-enabled PyTorch, `gguf`, nvcc and a C++ compiler.
Install PyTorch for your CUDA version first, then:

```sh
# From the root of the public home-llm checkout:
cd recipes/deepseek-orca
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install ninja
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The original run used8xRTX5090,512GB host RAM and about1.1TB disk. Routed conversion
first-to-last commit took9h18m including source/network/recovery overhead; trunk
and lookup processing overlapped. Final assembly/validation took about56minutes.
Those are historical observations, not timing promises. Output model shards total
332535797816bytes. Keep source/job artifacts plus another full candidate-sized
copy and at least24GiB reserve; finalization refuses insufficient disk. Source
range reads avoid a full weight download, but require reliable access. If weights
are locally cached, the reader verifies pinned LFS full SHA before trusting them.
Do an actual clean network speed test and small real HF transfer before renting
or moving hundreds of GB. This recipe creates no rentals or services.

This sanitized public adapter changes paths/control metadata, so private run
manifest hashes are not copied. `SOURCE_PROVENANCE.json` records the frozen
numerical sources; `CODE_SHA256.json` hashes this exact public version. Numerical
source is retained, but floating implementation/hardware and path/provenance
adaptation mean a new full run is not promised byte-identical to the published
entire pack. Never combine outputs from different encoder pins or numerical
policies in one work directory.

## 1. Local tests and command checks

```sh
# Public independent fixture required by the contract test (no model weights).
git clone https://github.com/beamivalice/sushi.git public-sushi
(cd public-sushi && git checkout d5e876edcb1538a347041b1b7cc28a0e6e8f0af6)
export SUSHI_SRC="$PWD/public-sushi"
python -m unittest discover -p 'test_*.py' -v
python fetch_trunk_calibration.py --self-check
python orca_export.py --help
python run_orca_conversion.py --help
```

Tests do not download model weights or run inference. CUDA tests and public
external fixtures are optional/explicit. For independent public fixture tests:
check out the pinned Sushi revision separately and set `SUSHI_SRC` to it.
Full large-model conversion is not rerun as part of publishing this recipe.

## 2. Prepare public calibration inputs and assets

Run from this directory; choose one writable work root:

```sh
export WORK="$PWD/work"
mkdir -p "$WORK"
python prepare_counts.py "$WORK/counts.json"
python fetch_trunk_calibration.py --output "$WORK/trunk.npz"
python orca_export.py "$WORK/assets" --prepare-assets
```

The calibration generator uses bounded public imatrix reads and resumable per-key
cache files. Expected337keys; grouped `wo_a` diagonals are `[8,4096]`. Its sidecar
lists52 skipped linears (40routers,8 unsupported grids,4 missing calibration).
Routers belong exclusively to assembled expert shards; other skipped linears
remain F32. Nonpositive/unvisited counts fail closed unless the explicit routed
identity-n28 policy applies. Original donor chat template/map are retained.

## 3. Pilot first: one projection, then one expert

```sh
python orca_export.py "$WORK/pilot-down" --layer 20 --expert 7 \
  --projection down --n 28 --backend cuda --batch 256 --imatrix
python orca_export.py "$WORK/pilot-expert" --layer 20 --expert 7 \
  --n 28 --backend cuda --batch 256 --imatrix
```

Inspect each manifest's source/encoder pins, finite SSE/weighted error, F16 scales
and exact bank shapes. Projection pilots are not loadable models. Their metrics
are quantization proxies, not general capability, abliteration or KLD scores.

## 4. Full routed experts

Start only after the pilot and resource/network checks pass. `--execute` is
explicit; without it the command emits a dry-run job plan only.

```sh
python run_orca_conversion.py "$WORK/routed" --catalog ./orca_256g_tensors.json \
  --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --batch 256 --n 28 \
  --counts-coverage "$WORK/counts.json" --unvisited-policy identity-n28 --execute
```

Persistent spawned workers cache headers/calibration counts. Each expert/projection
has an atomic independent manifest. Resume with exactly the same code, source,
parameters and directory; completed artifacts are SHA-checked. Root manifest lag
is recoverable from committed per-job manifests. Do not start another writer on
that work root. Do not use `--finalize-layers`: that older pilot assembly includes
dense trunk and would duplicate the standalone trunk stage below.

Optional measured rates JSON is `layer -> expert -> projection -> n`; supported
pilot rates are24/28/32/40. Do not invent sensitivity from rate alone. The released
candidate is uniform n28. Mixed groups require identical router permutations and
at most32 joint rate signatures, validated by the layout contract.

## 5. Trunk, globals and lookup tables

These use independent output directories and can run alongside routed encoding
when resource admission permits. The original trunk used an additional context on
GPU7; measure contention rather than assume this always improves throughput.

```sh
python run_trunk_conversion.py "$WORK/trunk" --calibration "$WORK/trunk.npz" \
  --provenance "$WORK/trunk.json" --catalog ./orca_256g_tensors.json \
  --gpu 7 --batch 256 --n 64 --execute
python prepare_globals.py "$WORK/globals-original" "$WORK/globals-v2"
python engram_export.py "$WORK/model-engram-01.safetensors" --layer 1 \
  --catalog ./orca_256g_tensors.json --backend cuda --gpu 0 --chunk-rows 131072
python engram_export.py "$WORK/model-engram-14.safetensors" --layer 14 \
  --catalog ./orca_256g_tensors.json --backend cuda --gpu 1 --chunk-rows 131072
```

`prepare_globals.py` creates269globals then a separate correction epoch:135
original F32 vectors are restored,134unchanged artifacts are hardlinked. Do not
use its first, all-vision-BF16 output as the final globals input.

The Engram commands above make bounded authenticated range reads without caching
whole donor shards. For faster repeated reads, download the pinned second GGUF
shard using your normal HF client into `CACHE/Mix-256G/<exact-source-filename>` and
pass `--local-source-dir "$CACHE"`. This costs110595290592bytes; the reader verifies
its pinned full LFS SHA before the first local read. Each lookup output is about
101GB; scale/weight share one shard. Rows are unequal384006168/384016682. Never
attempt unbounded F32 materialization. Chunk size/backend/code are manifest-pinned;
use the same values on resume. CUDA quantization has CPU torch oracle checks.

## 6. Experts-only assembly and final candidate

After all384x3projection jobs of a layer commit (all40for the command below):

```sh
python assemble_orca_experts.py "$WORK/routed/manifest.json" "$WORK/experts" \
  --layers 0-39 --catalog ./orca_256g_tensors.json
python materialize_orca_candidate.py "$WORK/candidate" \
  --experts-root "$WORK/experts" --trunk-root "$WORK/trunk" \
  --globals-root "$WORK/globals-v2" \
  --engram "$WORK/model-engram-01.safetensors" "$WORK/model-engram-14.safetensors" \
  --assets "$WORK/assets/assets.json" --catalog ./orca_256g_tensors.json
# Read the plan/missing report first, then repeat with --execute only if complete.
```

Finalization copies/restamps payloads into root `model*.safetensors`, synthesizes
actual codec ceilings/shared revision, writes a relative index and verifies exact
native coverage before declaring structural preflight pass. Mandatory source
checks include all committed artifact SHA and lookup chunk SHA. The independent
final report remains runtime_unverified/ready=false: a completed conversion is not
proof of inference or quality. No work items are deleted automatically.

## 7. Public repack, load and evaluation

On Apple Silicon, obtain and SHA-verify a compatible public Sushi build (the
released Mac smoke used1.6.0). After verifying the complete pack:

```sh
sushi repack "$WORK/candidate"
sushi repack "$WORK/candidate" --check
sushi serve --model "$WORK/candidate" --host 127.0.0.1 --port 12453 \
  --ctx-size 2048 --ssd-budget-gb 24 --no-mtp --no-vision \
  --prefill-chunk 512 --prefix-cache-entries 0 --prefix-cache-disk 0 \
  --max-resident-models 1 --max-tokens 256 --timeout 600 --no-update-check
```

The24GiB budget is a tested48GBMac configuration, not total process memory or a
universal safe setting. Preserve strict admission checks. Repack performs public
inverse-byte validation; it legitimately changes expert payload/index hashes.
Keep original download and post-repack proofs separately. No full model load,
network/inference benchmark or conversion is executed by the recipe tests.

## License and attribution

This recipe's original glue is MIT licensed; see `LICENSE`. Public Sushi consumer
codebook/packing and transform references retain their MIT notice in
`LICENSE-SUSHI`. `ATTRIBUTION.md` records source revisions and upstream notices.
Dependencies and model weights retain their own licenses. This directory does
not relicense the rest of this repository or redistribute donor weights,
private Sashimi sources, authentication material or user evaluation payloads.
