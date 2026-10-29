# AutoRound W4A16 quant of the JetBrains blend, dbirks recipe (compressed-tensors out).
# usage: CUDA_VISIBLE_DEVICES=<uuid> ar_quant.py --out DIR --calib pile|mix --bs 4 --gas 1 [--lowmem]
import argparse, json, os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import torch
from transformers import AutoTokenizer
from auto_round import AutoRound

SRC = "~/HyperQwen-0.32/models/Qwen3.8-3.6-27B-blend"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--calib", choices=["pile", "mix"], default="pile")
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--gas", type=int, default=1)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--lowmem", action="store_true")
a = ap.parse_args()
t0 = time.time()
cfg = json.load(open(f"{SRC}/config.json"))["text_config"]
tok = AutoTokenizer.from_pretrained(SRC)

# keep GatedDeltaNet recurrence-control projections in BF16 (dbirks recipe)
layer_config = {}
for i, t in enumerate(cfg["layer_types"]):
    if t == "linear_attention":
        for p in ("in_proj_a", "in_proj_b"):
            layer_config[f"model.language_model.layers.{i}.linear_attn.{p}"] = {"bits": 16}

SEQ, N = 2048, 128

def pack(texts, want, tag):
    """Concatenate texts into samples of >= SEQ tokens."""
    out, buf, n = [], "", 0
    for t in texts:
        buf += t + "\n\n"
        if len(buf) > SEQ * 2 and len(tok(buf).input_ids) >= SEQ + 16:
            out.append(buf); buf = ""
            if len(out) >= want:
                break
    print(f"calib {tag}: {len(out)} samples", flush=True)
    return out

if a.calib == "pile":
    dataset = "NeelNanda/pile-10k"
else:
    from datasets import load_dataset
    def ru():
        for r in load_dataset("IlyaGusev/saiga_scored", split="train", streaming=True):
            if r.get("language") == "Russian" and (r.get("opus_score") or 0) >= 8:
                msgs = [{"role": {"bot": "assistant"}.get(m["role"], m["role"]), "content": m["content"]} for m in r["messages"]]
                yield tok.apply_chat_template(msgs, tokenize=False)
    def code():
        for r in load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True):
            conv = r.get("conversations") or []
            msgs = [{"role": "user" if c["from"] == "user" else "assistant", "content": c["value"]} for c in conv]
            if msgs:
                yield tok.apply_chat_template(msgs, tokenize=False)
    def pile():
        for r in load_dataset("NeelNanda/pile-10k", split="train", streaming=True):
            yield r["text"]
    dataset = pack(ru(), 48, "ru") + pack(code(), 48, "code") + pack(pile(), 32, "pile")
    N = len(dataset)

ar = AutoRound(SRC, tokenizer=tok, scheme="W4A16", dataset=dataset, nsamples=N, seqlen=SEQ,
               batch_size=a.bs, gradient_accumulate_steps=a.gas, iters=a.iters,
               low_gpu_mem_usage=a.lowmem, low_cpu_mem_usage=True, device_map=0,
               trust_remote_code=True, quant_nontext_module=False, seed=42,
               layer_config=layer_config)
print(f"init {time.time()-t0:.0f}s", flush=True)
ar.quantize_and_save(output_dir=a.out, format="llm_compressor")
print(f"QUANT_DONE {a.out} total {time.time()-t0:.0f}s", flush=True)
