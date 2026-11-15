# Kiwi overnight production telemetry

Window: **15 November 2026, 00:00-08:30 CET (UTC+01:00)**. This is ordinary production traffic, not a controlled benchmark or a sustained-load test. No load was generated for this capture.

Source: the Prometheus time series used by Grafana, `query_range` at 60-second evaluation steps. The 511 sampled timestamps include both endpoints. Public data in [telemetry.json](telemetry.json) omit hostnames, IPs and GPU UUIDs.

- Live config checked on 15 November: Kiwi Maya-XS abliterated pack; Maya v1.1.61 with PR #959 deltas and PR #911, local engine commit `56154a5eccc6b4eecfc12b32a622634fe184d2cf`; `DS4_IQ1S512=1`, split `16,8,16`, 524288 context, DSpark, parallel 2. Runtime refusal-direction ablation is off by default (separate from the pack).
- Decode: **69.0 tokens/s median of 81 non-zero sampled gauge values**; sampled peak 147.5 tokens/s. The gauge is a 15-second non-zero mean of Maya's 5-second active-request rates, summed across requests. This is neither a per-request benchmark nor a whole-night average.
- Prefill: **3,764 tokens/s median of 19 non-zero sampled gauge values**. Production prompts include cache reuse: this is NOT a cold-prefill benchmark.
- Completed output-token counter increased by **258,665 tokens** between the endpoints. This accounts for completed requests; it is not an instantaneous decode-rate measurement.
- Exporter engine-up gauge was 1 at **511/511 sampled timestamps**. One-minute sampling does not rule out shorter outages. Mostly idle traffic is visible in the chart.
- Temperature/power are sampled GPU sensors (NVML), not wall power. Sampled maxima are not continuous-time maxima.

Queries: `rig_llm_decode_tokens_per_second{lane="DS"}`, `rig_llm_prefill_tokens_per_second{lane="DS"}`, `rig_llm_requests_running{lane="DS"}`, `rig_llm_up{lane="DS"}`, `rig_llm_generation_tokens_total{lane="DS"}`, `nvidia_smi_temperature_gpu`, `nvidia_smi_power_draw_watts`.
