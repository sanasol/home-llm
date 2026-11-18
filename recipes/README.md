# Sushi conversion recipes

These recipes convert **existing community-abliterated weights** into Sushi-compatible packs. I did not perform the original abliteration. Credit for the weight edits belongs to the donor authors; credit for Sushi and its public formats belongs to the upstream project.

| Recipe | What it builds | Scope |
|---|---|---|
| [DeepSeek Heretic](deepseek-heretic/README.md) | Official Sushi STQ pack with the 38 donor-edited attention-output banks re-encoded | Experts remain official; not a full Heretic requantization |
| [DeepSeek Orca](deepseek-orca/README.md) | Full donor-GGUF-derived PAIR1/SPEC1/FP8 Sushi pack | Re-encodes the actual OrcaRouter weights, including experts; adds requantization loss |

Get the scripts:

```sh
git clone https://github.com/sanasol/home-llm.git
cd home-llm
# Choose one recipe, then follow its own environment and pilot instructions:
cd recipes/deepseek-heretic
# Or: cd recipes/deepseek-orca (from the repository root).
```

Start with the recipe's small pilot and tests, not the full conversion. Downloading a ready model is much easier than rebuilding one. These are model-specific research recipes, not a universal one-command converter or the private Sashimi encoder.

## Timings and hardware

The released full Orca build used eight RTX 5090 GPUs. Expert projections took **9 h 18 min** wall-clock, trunk **8 h 40 min** in parallel, and final assembly/checks another **56 min**. Those numbers include substantial network and orchestration overhead, not pure GPU time. The 110.6 GB Engram donor shard took **6 h 42 min including verification** on that slow rental. On a later well-connected machine the entire 348.9 GB source set downloaded and verified in **15 min 38 s**. Source downloads and conversion stages overlap; do not add these durations together.

An experimental GPU-resident rewrite later passed small numerical/performance tests, but its full reserve run was cancelled and its outputs were not released. The published original Orca weights must not be described as produced by that rewrite. No complete, independently rerun timing is available here for the smaller Heretic-only build.

Before large rented jobs, run a clean network speed test plus a small actual Hugging Face upload/download. Advertised host bandwidth is not a benchmark. Check the recipe's disk/RAM requirements and storage/transfer charges first.

## Limits

- Abliteration, quantization, inference support and evaluation are separate steps. These scripts address conversion, not training a refusal-removal method.
- A new architecture or release such as Qwen4 Flash needs explicit engine, tensor-mapping and quantizer support. Do not assume changing a model ID is sufficient.
- Numerical reconstruction checks and short refusal pilots do not establish broad capability preservation. Follow the model cards for the measured scope.
- No model weights, tokens, private endpoints, raw harmful test outputs or automatic rented-instance provisioning are included.
