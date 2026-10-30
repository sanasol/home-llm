# 2026-10-26 · Steam Deck + 5090 · NInfer: KV dtype and DFlash2 draft window

Same host and engine as
[2026-10-25-deck-5090-tensorfold-vs-ninfer](../2026-10-25-deck-5090-tensorfold-vs-ninfer/README.md):
NInfer `64d10fbf`, Qwen3.8-27B NVFP4. Common flags: `--vision`, 2 streams,
`--device-state-slots 2`, `--kv-capacity auto`, `--lm-head-draft`,
`--host-context-mib 4096`. Servers ran on `127.0.0.1:18080` while `ninfer.service`
was stopped.

Scripts: [exp.sh](exp.sh), [exp2.sh](exp2.sh), [bench.py](bench.py), [needle.py](needle.py).
Raw logs: [run1.log](run1.log), [run2.log](run2.log).
- `bench.py`: greedy, thinking off. Short prompts (code / prose / list) = median of 3.
- `tool_call`: the reply is a `write_file` call containing a whole code file,
  `tool_choice: required`. NInfer doesn't stream tool-call arguments, so speed here
  = completion_tokens / full request time.
- Long prompts: Zig sources with a unique marker — prompt cache never hits.
- `needle.py`: unique code at depth 10/50/90%; the answer must quote it verbatim.

## Results (tok/s)

| | fp8 K=5 | **fp8 K=7** (was) | fp8 K=9 | **k8v4 K=7** |
|---|---|---|---|---|
| KV pool, tokens | 215,808 | 215,488 | 215,232 | **276,672** |
| max-context w/ vision | 212,992 | 212,992 | 212,992 | **240,000** (scheduler fits 262,144 too) |
| code | 236 | 257 | 254 | **274** |
| prose | 147 | 163 | 161 | **181** |
| list | 161 | **201** | 195 | 175 |
| tool_call (file via `write_file`) | 256 | 325 | **361** | 328 |
| 8.8K: prefill / decode | 10,024 / 174 | 10,043 / 204 | 10,047 / **213** | 10,054 / 202 |
| 35K: prefill / decode | 9,358 / 159 | 9,369 / 169 | 9,369 / **192** | 9,377 / 182 |
| 108K: prefill / decode | 6,502 / 161 | 6,508 / 166 | 6,508 / 148 | 6,519 / **172** |
| needle, 3 depths | — | 3/3 at 131K | — | 3/3 at 131K and **3/3 at 162K** |

## Conclusions

- **`k8v4` beats `fp8`:** KV pool +28%, 240K per request with vision. Decode on code,
  prose and long context equal or faster; prefill unchanged. Needle at 162K found at
  all three depths. Only the "list" workload regresses (175 vs 201). I switched.
- **K=5 is worse than K=7 everywhere.** A higher draft-acceptance rate at a small
  window does not translate into speed.
- **K=9 vs K=7 on fp8:** tool_call +11%, faster at 8–35K (+4…+14%), slower at 108K
  (−11%). If your agent keeps context under ~75K, K=9 may win. Not tested on k8v4.
