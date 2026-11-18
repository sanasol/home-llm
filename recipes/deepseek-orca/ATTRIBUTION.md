# Attribution and provenance

- Public Sushi: https://github.com/beamivalice/sushi at
  d5e876edcb1538a347041b1b7cc28a0e6e8f0af6. Codebooks, public tile permutation,
  consumer equations and pack validation informed the included independently
  tested reference/encoder. MIT copyright2026 Theinruj Toranavikrai and David
  Dalcu is retained in LICENSE-SUSHI. No private Sashimi converter is included.
- Sushi's MIT code includes inherited mlx-serve portions. Upstream distribution
  also contains separately licensed third-party software; this recipe does not
  bundle that engine/dependency distribution. Obtain its complete LICENSE/NOTICE
  when building/downloading Sushi itself.
- llama.cpp PR30817 head b72050d03ae14bb3cb5138ecefab3f3f65157d31:
  GGUF-to-native mapping and fused Engram product behavior were checked against
  public conversion/deepseek41.py and runtime source. Installed gguf handles
  reference GGUF dequantization; its own license remains applicable.
- Donor and base model licenses/attributions must accompany any redistributed
  weights. This source-only directory contains no model weights.

SOURCE_PROVENANCE.json records the exact originally frozen hybrid CPU/CUDA
numerical module bytes and identifies public stage adapters. Public portability
edits: bundled-relative default catalog path, local calibration output default,
removal of workstation-specific test command paths, and HTTPS-only redirects that
strip Authorization on cross-origin CDN redirects while preserving Range. Those adapters have new
CODE_SHA256.json identities; no private-run manifests are fabricated for them.

Later GPU-resident math prototypes and their cancelled reserve artifacts are
not used to describe the released model and are not included in this recipe.
