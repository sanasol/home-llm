"""rig_affinity: per-conversation stickiness key for the 2x3090 pool (model group qwen3.8-27b).

Why: LiteLLM `session_affinity` pins on metadata.session_id (header x-litellm-session-id). Hermes
derives that value from the conversation ROOT, so every subagent of one parent shares it — a
parallel fan-out would all pin to ONE card. This hook makes the key
    <client session id or "-">|sha256(first user message)[:16]
so: turns of one conversation keep one key (prefix cache stays warm on one card), sibling
subagents (different goals) get different keys and spread via least-busy, and two separate
chats with identical first prompts still differ by their Hermes session id.
After a compaction the first user message changes -> new key -> may move once (its prefix
cache is mostly invalid at that point anyway).

Loaded from /opt/litellm (config dir): litellm_settings.callbacks: ["rig_affinity.handler"].
Stdlib + litellm only. Never raises into the request path.
"""
import hashlib

from litellm.integrations.custom_logger import CustomLogger

POOL = {"qwen3.8-27b"}
try:
    from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY as GENERATED
except Exception:  # noqa: BLE001
    GENERATED = "litellm_session_id_generated"


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class RigAffinity(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            if data.get("model") not in POOL:
                return data
            first = next((_text(m.get("content")) for m in data.get("messages") or []
                          if isinstance(m, dict) and m.get("role") == "user"), "")
            if not first:
                return data
            digest = hashlib.sha256(first[:8000].encode("utf-8", "ignore")).hexdigest()[:16]
            metas = [d for d in (data.get("metadata"), data.get("litellm_metadata")) if isinstance(d, dict)]
            if not metas:
                data["metadata"] = {}
                metas = [data["metadata"]]
            client_sid = next((str(m["session_id"]) for m in metas
                               if m.get("session_id") and not m.get(GENERATED)), "-")
            key = f"{client_sid}|{digest}"
            for m in metas:
                m["session_id"] = key
                m.pop(GENERATED, None)
        except Exception:  # noqa: BLE001 - affinity is best-effort
            pass
        return data


handler = RigAffinity()
