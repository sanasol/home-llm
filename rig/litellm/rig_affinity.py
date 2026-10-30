"""rig_affinity: per-conversation stickiness + liveness filter for the rig's GPU pool.

Pool today: `qwen3.8-flash-next-3090` ("FN 3090") = two Strata lanes on the 3090s (:8003, :8004),
one shared expert arena (docs/23).

1) Stickiness key. LiteLLM `session_affinity` pins on metadata.session_id (header
   x-litellm-session-id). Hermes derives that value from the conversation ROOT, so every subagent of
   one parent shares it — a parallel fan-out would all pin to ONE card. This hook makes the key
       <client session id or "-">|sha256(first user message)[:16]
   so: turns of one conversation keep one key (the lane's resident KV / conversation cache stays
   warm), sibling subagents (different goals) get different keys and spread via least-busy, and two
   separate chats with identical first prompts still differ by their Hermes session id.
   After a compaction the first user message changes -> new key -> may move once.

2) Liveness filter. LiteLLM background health checks send a real completion; on a 1-slot Strata
   lane that would queue behind agent work and evict the lane's resident conversation every 15 s.
   Instead `async_filter_deployments` drops pool deployments whose engine `GET /health` is not
   {"status":"ok","loaded":true} (cached LIVENESS_TTL s, 1.5 s timeout). If every member looks
   dead the list is returned unchanged (the request then fails loudly instead of silently).

Loaded from /opt/litellm (config dir): litellm_settings.callbacks: ["rig_affinity.handler", ...].
Stdlib + litellm (+ httpx, a litellm dependency) only. Never raises into the request path.
"""
import hashlib
import time

import httpx
from litellm.integrations.custom_logger import CustomLogger

POOL = {"qwen3.8-flash-next-3090"}
LIVENESS_TTL = 5.0
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


def _health_url(deployment):
    base = ((deployment or {}).get("litellm_params") or {}).get("api_base") or ""
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/health" if base else None


class RigAffinity(CustomLogger):
    def __init__(self):
        super().__init__()
        self._live = {}  # url -> (checked_at, ok)

    async def _alive(self, url):
        now = time.monotonic()
        hit = self._live.get(url)
        if hit and now - hit[0] < LIVENESS_TTL:
            return hit[1]
        ok = False
        try:
            async with httpx.AsyncClient(timeout=1.5) as c:
                r = await c.get(url)
            j = r.json() if r.status_code == 200 else {}
            ok = j.get("status") == "ok" and j.get("loaded", True) is not False
        except Exception:  # noqa: BLE001 - unreachable = dead
            ok = False
        self._live[url] = (now, ok)
        return ok

    async def async_filter_deployments(self, model, healthy_deployments, messages,
                                       request_kwargs=None, parent_otel_span=None):
        try:
            if model not in POOL or not healthy_deployments:
                return healthy_deployments
            alive = []
            for d in healthy_deployments:
                url = _health_url(d)
                if url is None or await self._alive(url):
                    alive.append(d)
            return alive or healthy_deployments
        except Exception:  # noqa: BLE001 - routing must never break on the filter
            return healthy_deployments

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
