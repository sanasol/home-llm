# Refusal lab

Small reproducible HarmBench/XSTest API runs and an offline comparison viewer. Python standard library only; no pip install. Generated answers are saved as data, never executed. No prefix-based refusal scoring or official HarmBench judge.

## Setup on Mac

```bash
git pull
cd eval/refusal
cp .env.example .env
chmod 600 .env
open -e .env
```

Fill your own `RIG_API_KEY` (client LiteLLM key) and `DEEPSEEK_API_KEY`. Sushi normally needs no key on loopback. `.env`, dataset cache and results are gitignored. Shell environment overrides `.env`. Values are literal: no shell interpolation, expansion or commands. Keep API keys out of URLs. Do not serve this directory with `python -m http.server`: it contains `.env`.

Profiles:

| Profile | Model / endpoint | Reasoning control |
| --- | --- | --- |
| sushi | heretic, loopback12453 | chat_template_kwargs, chat/thinking off |
| rig | your `RIG_MODEL` at `RIG_URL`; default loopback8010 | template chat; confirm actual response reasoning |
| deepseek | deepseek-flash, api.deepseek.com | thinking.type=disabled |

Direct API calls cost money. Rig requests consume a live inference slot. The runner does not start, stop or switch servers; Sushi must already be serving the selected model. No automatic fallback to another profile/model.

## Sample at most two standard behaviors per category

```bash
python3 run.py --profile sushi --dataset harmbench --per-category 2
```

This downloads the public CSV into `data/` if absent, hashes it, writes a human-readable selection and prints counts. It sends **no model requests without `--execute`**. Inspect the selection first. HarmBench contextual/multimodal tasks are excluded because they require different input construction.

```bash
caffeinate -i python3 run.py --profile sushi --dataset harmbench --per-category 2 --execute
```

Default output: `results/sushi-harmbench-256.jsonl` and its `.selection.json`. Start small with `--limit 2`. For all standard cases use `--per-category 0`. Change `--max-tokens 192` if needed; use a fresh output for changed settings. At1-2 tokens/s, long answers take minutes and a category sample can take tens of minutes.

### EXACT same selection on rig and direct API

```bash
caffeinate -i python3 run.py --profile rig \
  --selection results/sushi-harmbench-256.selection.json \
  --out results/rig-harmbench-256.jsonl --execute

caffeinate -i python3 run.py --profile deepseek \
  --selection results/sushi-harmbench-256.selection.json \
  --out results/api-harmbench-256.jsonl --execute
```

Use the same token budget in all three runs. `--selection` reuses exact prompts/IDs even if the upstream dataset changes. The chosen API payload must match each backend: use `--thinking-mode template|deepseek|none` or `RIG_THINKING_MODE` in `.env` if needed. A successful HTTP response does NOT prove reasoning was disabled; `reasoning_present` records returned reasoning fields. Don't compare a reasoning-truncated response as if it were a final answer.

### XSTest

```bash
caffeinate -i python3 run.py --profile sushi --dataset xstest \
  --labels all --per-category 2 --out results/heretic-xstest.jsonl --execute
```

`--labels safe` checks over-refusal; `--labels unsafe` selects contrast controls. Category sampling is by XSTest `type`; HarmBench sampling is by `SemanticCategory`. No jailbreak wrappers or automatic execution of responses.

### Resume and errors

Repeat the exact command: completed responses are skipped, failed requests retried. Network/API error is saved and the run stops instead of silently switching models. Mismatched settings or malformed existing JSONL fail closed. `finish_reason=length` is truncated/inconclusive, not successful compliance. `stop` only means generation finished. Do not confuse lack of private information with a normative safety refusal.

## Interactive viewer / comparison

```bash
open viewer.html
```

Drop/select multiple JSONL files, including earlier `heretic-256.jsonl` and category-run files. No HTTP server or package installation; files stay in your browser. Compare matched case IDs/prompts side-by-side, filter/search, inspect full answer and reasoning separately, assign manual labels and export annotations. Dataset/prompt incompatibilities must be reviewed rather than merging unrelated results. A model agreeing in its first sentence is not proof of substantive completion.

Manual labels are local browser storage, not benchmark ground truth. Export annotations before switching browsers/clearing storage. The viewer displays answers as text, not executable HTML/Markdown. Do not publish results containing sensitive test material unintentionally.

## Two NanoGPT judges from the UI

```bash
python3 serve.py
open http://127.0.0.1:8765/
```

Import your existing JSONL files, filter the rows you want assessed, enter your NanoGPT key in the password field and explicitly start judging. Alternatively set `NANOGPT_API_KEY` in `.env` for the local server. The key is not written to browser storage, JSONL or exports. The local server exposes ONLY the viewer and the judge POST endpoint; `.env`, arbitrary files and result directories are never served. Bind is loopback only. Do not expose/forward this server to untrusted clients.

The two requested judges are `abliteration-ai/abliterated-model-large-v2` and `deepseek/deepseek-v4.1-flash-1017-uncensored:thinking`, via `https://api.nano-gpt.com/api/v1/chat/completions`. Each selected row costs up to two API requests. The UI runs4 requests concurrently by default, adjustable from1 to16. Both judges and different rows share the bounded queue. Stop aborts active browser requests and prevents further queue dispatch, but upstream requests already accepted may still be billed. No automatic retries on rate limits. Prompts and final answers leave your computer and are sent to NanoGPT; inspect them and consent before starting. No request happens just from importing a file. No automatic endpoint fallback or paid retry.

Judge labels: compliance (passed), refusal/safe_redirection (not fulfilled), inconclusive and errors separately. This is a model judge's opinion, NOT an official HarmBench score; disagreements are shown rather than silently merged. Empty finals are inconclusive without an API call. The thinking judge has an8192-token output/reasoning budget; the other judge has2048. Exhausted judge budgets, missing final verdicts and malformed JSON count as judge errors, not model inconclusive answers. The UI recognizes the exact legacy budget/JSON-error reasons in old exports too. Use the error-only retry button to assess only failed judge/answer pairs: successful assessments are retained, missing assessments are not silently added, and no model answers are regenerated. Retries are explicit paid API requests. Model truncation is displayed separately; a valid judge can still assign inconclusive when the evaluated model answer is incomplete. You cannot infer harmful capability merely from fewer refusals. Judge output may be wrong or susceptible to prompt injection despite the data-only system instruction; review uncertain cases manually.

If the Python.org macOS installer reports `CERTIFICATE_VERIFY_FAILED`, run `python3 -m pip install certifi` using the SAME interpreter that starts `serve.py`, then restart it. The proxy uses an explicit `NANOGPT_CA_BUNDLE`/`SSL_CERT_FILE` if set, otherwise certifi when installed, otherwise the macOS system PEM bundle. Never disable certificate verification. After any `.env` change restart `serve.py`. API errors are displayed per judge row, including upstream HTTP status; HTTP401 is an API-authentication error, not TLS.

Export the judge results before closing/clearing the browser. New exports use `refusal-lab.judges.v2` and include source filename, tested model, case ID/category, dataset hash, original prompt, final answer, finish reason and generation settings per assessment. Importing v2 restores the answer cards and per-run summaries even without the original JSONLs; no API calls are sent. Older v1 exports remain supported: load the original JSONLs alongside them, then re-export to enrich their matching identities without paying for a re-evaluation. A v1 export alone cannot recover missing model/case metadata. Imported judge reason/raw/verdict remain opinions, not new model responses. The viewer opened directly as `file://` still works for local inspection but judging requires this loopback server. The backend accepts only the fixed judge models; it never executes evaluated answers or generated code.

## Checks

```bash
python3 -m unittest discover -p 'test_*.py' -v
```

Tests use a local mock API, not paid/live model endpoints, and verify category selection, literal `.env`, provider-specific payloads, resume/settings isolation and omission of the API key from saved results.

Sources: [HarmBench](https://github.com/centerforaisafety/HarmBench), [XSTest](https://github.com/paul-rottger/xstest). Small stratified samples are not official full benchmark scores or proof of retained model capabilities.
