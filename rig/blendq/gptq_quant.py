# Fast-fail lane: llm-compressor GPTQ W4A16 g128 of the blend (compressed-tensors out).
import os, sys, time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from llmcompressor.utils import load_context
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

SRC = "~/HyperQwen-0.32/models/Qwen3.8-3.6-27B-blend"
OUT = sys.argv[1]
SEQ, N = 2048, 128
t0 = time.time()
tok = AutoTokenizer.from_pretrained(SRC)
with load_context(Qwen3_5ForConditionalGeneration, load_mtp=False):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        SRC, dtype=torch.bfloat16, device_map="auto_offload", offload_folder="~/blendq/offload_gptq")
print(f"loaded {time.time()-t0:.0f}s", flush=True)

ds = load_dataset("NeelNanda/pile-10k", split="train").shuffle(seed=42).select(range(N))
ds = ds.map(lambda r: tok(r["text"], max_length=SEQ, truncation=True, add_special_tokens=False),
            remove_columns=ds.column_names)

recipe = GPTQModifier(targets="Linear", scheme="W4A16",
                      ignore=["lm_head", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*mtp.*"])
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=SEQ, num_calibration_samples=N,
        sequential_targets=["Qwen3_5DecoderLayer"])
print(f"quantized {time.time()-t0:.0f}s", flush=True)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print(f"QUANT_DONE {OUT} total {time.time()-t0:.0f}s", flush=True)
