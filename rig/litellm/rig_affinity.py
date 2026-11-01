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
import json
import os
import time

import httpx
from litellm.integrations.custom_logger import CustomLogger

POOL = {"qwen3.8-flash-next-3090"}
LIVENESS_TTL = 5.0
# per-request attribution log for the rig monitor: one JSON line per completion
# {t, sid, conv, model, url}. sid = the affinity key ("client|digest") or the raw
# client session id; conv = sha1(first user message)[:16] = the auto-<conv>.bin key the
# Strata serve layer parks conversations under, which lets the monitor join engine
# request rows with disk-restore events. Best-effort: any failure skips the line.
REQLOG = os.environ.get("RIG_REQLOG", "/var/lib/litellm/reqlog.jsonl")
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

    @staticmethod
    def _call_data(kwargs):
        """LiteLLM passes `model_call_details` itself as kwargs to async_log_success_event:
        messages sit on top, the chosen deployment's api_base and the request metadata under
        litellm_params, model_info also carries api_base. Older layouts nested it one level in."""
        if isinstance(kwargs, dict):
            if "messages" in kwargs or "litellm_params" in kwargs:
                return kwargs
            for k in ("model_call_details", "litellm_call_info"):
                d = kwargs.get(k)
                if isinstance(d, dict):
                    return d
        return {}

    @staticmethod
    def _api_base(kwargs, data):
        for d in (data.get("litellm_params"), data.get("model_info"), kwargs.get("model_info"), data):
            u = (d or {}).get("api_base") if isinstance(d, dict) else None
            if isinstance(u, str) and u.startswith("http"):
                return u
        return ""

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """One JSONL line per completion for the rig monitor (session id + conversation key)."""
        try:
            data = self._call_data(kwargs)
            msgs = data.get("messages") or []
            first = next((_text(m.get("content")) for m in msgs
                          if isinstance(m, dict) and m.get("role") == "user"), "")
            conv = hashlib.sha1(first.encode("utf-8", "replace")).hexdigest()[:16] if first else ""
            lp = data.get("litellm_params") or {}
            meta = lp.get("metadata") or data.get("metadata") or kwargs.get("metadata") or {}
            sid = str(meta.get("session_id") or "") if isinstance(meta, dict) else ""
            try:
                t = end_time.timestamp()
            except AttributeError:
                t = time.time()
            line = {"t": round(t, 3), "sid": sid, "conv": conv,
                    "model": str(data.get("model") or lp.get("model") or kwargs.get("model") or ""),
                    "url": self._api_base(kwargs, data)}
            d = os.path.dirname(REQLOG)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(REQLOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 - logging must never break the response path
            pass


handler = RigAffinity()
