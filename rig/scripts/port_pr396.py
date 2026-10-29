#!/usr/bin/env python3
"""Port HyperQwen PR #396 (gdn-ba-pad-small-m, cut against vLLM 0.31) onto the vLLM 0.32 venv.

envs.py hunks apply as-is via `patch`; the two qwen_gdn_linear_attn.py hunks that fail
on 0.32 are re-anchored here by exact-string replacement (each anchor must match once).
Writes staged copies next to the originals (*.pr396); install = copy over with backup.

usage: port_pr396.py VLLM_PKG_DIR
"""
import sys
from pathlib import Path

V = Path(sys.argv[1])
F = V / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
src = F.read_text()


def once(text, old, new):
    assert text.count(old) == 1, f"anchor matched {text.count(old)}x: {old[:60]!r}"
    return text.replace(old, new)


src = once(src, "    RowParallelLinear,\n)\n",
           "    RowParallelLinear,\n    UnquantizedLinearMethod,\n)\n")
src = once(src,
           "        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)\n",
           "        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)\n"
           "        # Pad 2-4 token batches of the bf16 in_proj_ba GEMM to 5 rows (see\n"
           "        # gdn_ba_proj below). Only the plain unquantized, bias-free projection.\n"
           "        self.ba_proj_pad_small_m = (\n"
           "            envs.VLLM_GDN_BA_PAD_SMALL_M\n"
           "            and current_platform.is_cuda()\n"
           "            and isinstance(self.in_proj_ba.quant_method, UnquantizedLinearMethod)\n"
           "            and self.in_proj_ba.bias is None\n"
           "        )\n")
src = once(src,
           "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
           "        ba, _ = self.in_proj_ba(hidden_states)\n\n"
           "        use_fused_gdn_spec_decode = (\n",
           "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
           "        if self.ba_proj_pad_small_m:\n"
           "            ba = torch.ops.vllm.gdn_ba_proj(hidden_states, self.in_proj_ba.weight)\n"
           "        else:\n"
           "            ba, _ = self.in_proj_ba(hidden_states)\n\n"
           "        use_fused_gdn_spec_decode = (\n")
src += '''

def gdn_ba_proj(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """in_proj_ba (bf16, N = 2 * num_v_heads) with 2-4 token batches padded to 5.

    cuBLAS picks a 16x16 wmma kernel for M = 2..4 at this skinny shape (~5x slower
    on an RTX 3090); MTP-3 verify steps (4 tokens) sit exactly in that window.
    A custom op so the branch on the batch size stays out of the compiled graph.
    """
    num_tokens = hidden_states.size(0)
    if 2 <= num_tokens <= 4:
        padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, 5 - num_tokens))
        return torch.nn.functional.linear(padded, weight)[:num_tokens]
    return torch.nn.functional.linear(hidden_states, weight)


def gdn_ba_proj_fake(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.size(0), weight.size(0)))


direct_register_custom_op(
    op_name="gdn_ba_proj",
    op_func=gdn_ba_proj,
    fake_impl=gdn_ba_proj_fake,
)
'''
assert "direct_register_custom_op" in src.split("def gdn_ba_proj(")[0], "direct_register_custom_op not imported"
Path(str(F) + ".pr396").write_text(src)
print("staged", str(F) + ".pr396")
