"""rig_affinity: per-conversation stickiness + liveness filter for the rig's GPU pool.

Pool today: `qwen3.8-flash-next-3090` ("FN 3090") = two Strata lanes on the 3090s (:8003, :8004),
one shared expert arena (docs/23).

1) Stickiness key. LiteLLM `session_affinity` pins on metadata.session_id (header
   x-litellm-session-id). Hermes derives that value from the conversation ROOT, so every subagent of
   one parent shares it — a parallel fan-out would all pin to ONE card. This hook makes the key
       <client session id or "-">|sha256(first user message)[:16]
   so: turns of one conversation keep one key (the lane's resident KV / conversation cache stays
   warm), sibling subagents (different goals) get different keys and spread, and two
   separate chats with identical first prompts still differ by their Hermes session id.
   After a compaction the first user message changes -> new key -> may move once.

2) Placement policy (06.11, load-based). The router's `least-busy` strategy only counts requests
   IN FLIGHT in the assignment instant: an idle-looking lane may be about to receive a chatty
   agent, ties break deterministically, and conversations pile onto one 3090 while the other
   idles. Counting PINNED conversations (the 04.11 policy) fixed the idle case but broke the
   opposite one: one-shot auxiliary calls (title/memory requests) pin a lane for PIN_TTL and
   count the same as a 30-turn agent, so a parallel fan-out put BOTH agents on one lane
   (observed 13:39-13:48 06.11: two agents queueing on :8004, running=1 waiting=1, full
   re-prefill every switch, while :8003 idled). So placement is by measured LOAD:
       score(lane) = pool requests routed there within ARRIVAL_WINDOW (recorded on success)
   a NEW key goes to the lane with the lowest score (tie -> round-robin). Pins still live PIN_TTL
   (stickiness unchanged), and a CONGESTION REBALANCER moves a conversation when its pinned lane
   holds >= 2 recently-active pins and another lane holds at least 2 fewer: the newest active pin
   on the busy lane re-places (one re-prefill, then resident again), with MOVE_COOLDOWN per key
   and MIN_MOVE_GAP globally so a fan-out cannot ping-pong. Converges to active-pin diff <= 1.
   `async_filter_deployments` narrows the deployment list to the pinned lane, so the hook is the
   single source of truth for placement; `session_affinity` stays as a no-op mirror.
   The map is refreshed from BOTH the filter and `async_pre_call_hook` (session_affinity can
   short-circuit before the filter runs, which would otherwise let pins age out under active chats).

3) Liveness filter. LiteLLM background health checks send a real completion; on a 1-slot Strata
   lane that would queue behind agent work and evict the lane's resident conversation every 15 s.
   Instead `async_filter_deployments` drops pool deployments whose engine `GET /health` is not
   {"status":"ok","loaded":true} (cached LIVENESS_TTL s, 1.5 s timeout). If every member looks
   dead the list is returned unchanged (the request then fails loudly instead of silently). A
   conversation whose pinned lane died is re-placed to the least-pinned live lane.

Loaded from /opt/litellm (config dir): litellm_settings.callbacks: ["rig_affinity.handler", ...].
Stdlib + litellm (+ httpx, a litellm dependency) only. Never raises into the request path.
"""
import hashlib
import json
import os
import time
from collections import deque

import httpx
from litellm.integrations.custom_logger import CustomLogger

POOL = {"qwen3.8-flash-next-3090"}
LIVENESS_TTL = 5.0
# pin map entry lifetime; sliding (refreshed every turn), mirrors deployment_affinity_ttl_seconds
PIN_TTL = 3600.0
# load window: pool requests per lane counted over this window decide placement
ARRIVAL_WINDOW = 300.0
# a pin counts as "active" for rebalancing when seen within this window
ACTIVE_WINDOW = 180.0
# rebalance guards: per-key cooldown between moves, and a global minimum gap so a
# parallel fan-out moves one conversation at a time instead of stampeding
MOVE_COOLDOWN = 600.0
MIN_MOVE_GAP = 30.0
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
        self._live = {}       # url -> (checked_at, ok)
        self._pins = {}       # affinity key -> (lane health-url, last_seen monotonic)
        self._rr = 0          # round-robin counter for assignment ties
        self._arrivals = {}   # lane health-url -> deque of routed-request completion times
        self._moved_at = {}   # affinity key -> last rebalance move (monotonic)
        self._last_move = 0.0

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

    # ---- pin map (sync: single asyncio loop, no awaits inside) ----

    def _prune(self, now):
        for k in [k for k, (_, seen) in self._pins.items() if now - seen > PIN_TTL]:
            self._pins.pop(k, None)
        for dq in self._arrivals.values():
            while dq and now - dq[0] > ARRIVAL_WINDOW:
                dq.popleft()
        for k in [k for k, t in self._moved_at.items() if now - t > PIN_TTL]:
            self._moved_at.pop(k, None)

    def _touch(self, key, url):
        if key and url:
            self._pins[key] = (url, time.monotonic())

    def _score(self, urls, now):
        """Load proxy: requests routed to each lane within ARRIVAL_WINDOW (recorded on
        success, so it counts real traffic, not just pin bookkeeping)."""
        return {u: sum(1 for t in self._arrivals.get(u, ()) if now - t <= ARRIVAL_WINDOW)
                for u in urls}

    def _active(self, urls, now):
        """Pins seen within ACTIVE_WINDOW per lane: active conversations, ghosts excluded."""
        act = {u: 0 for u in urls}
        for u, seen in self._pins.values():
            if u in act and now - seen <= ACTIVE_WINDOW:
                act[u] += 1
        return act

    def _place(self, key, urls):
        """Pick (or keep) the lane for `key` among `urls`; refresh the pin."""
        now = time.monotonic()
        self._prune(now)
        hit = self._pins.get(key)
        if hit and hit[0] in urls:
            lane = hit[0]
            act = self._active(urls, now)
            # congestion rebalance: pinned lane holds several recently-active conversations
            # while another lane holds >= 2 fewer -> move this pin if it is the newest active
            # one there (the least invested conversation pays the single re-prefill). Guards:
            # per-key MOVE_COOLDOWN + global MIN_MOVE_GAP stop ping-pong after a fan-out.
            if (act[lane] >= 2
                    and now - self._moved_at.get(key, 0.0) > MOVE_COOLDOWN
                    and now - self._last_move > MIN_MOVE_GAP):
                mine = max((k for k, (u, s) in self._pins.items()
                            if u == lane and now - s <= ACTIVE_WINDOW),
                           key=lambda k: self._pins[k][1], default=None)
                cands = sorted(u for u in urls if act[lane] - act[u] >= 2)
                if cands and mine == key:
                    sc = self._score(urls, now)
                    lane = min(cands, key=lambda u: (sc[u], cands.index(u)))
                    self._moved_at[key] = now
                    self._last_move = now
            self._pins[key] = (lane, now)
            return lane
        # new key: the lane with the lowest recent request rate; tie -> round-robin
        sc = self._score(urls, now)
        m = min(sc.values())
        tied = sorted(u for u in urls if sc[u] == m)
        self._rr += 1
        lane = tied[self._rr % len(tied)]
        self._pins[key] = (lane, now)
        return lane

    @staticmethod
    def _key_from(request_kwargs):
        """The affinity key pre_call_hook wrote into metadata ("client|digest").

        A raw/generated session id never contains "|", so this only matches our key;
        anything else -> None -> fall back to the router's own choice."""
        for k in ("metadata", "litellm_metadata"):
            m = (request_kwargs or {}).get(k)
            if isinstance(m, dict):
                sid = m.get("session_id")
                if isinstance(sid, str) and "|" in sid:
                    return sid
        return None

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
            alive = alive or healthy_deployments
            key = self._key_from(request_kwargs)
            if key is None:
                return alive
            urls = [_health_url(d) for d in alive]
            if any(u is None for u in urls):
                return alive
            if len(alive) == 1:
                self._touch(key, urls[0])
                return alive
            url = self._place(key, urls)
            chosen = [d for d in alive if _health_url(d) == url]
            return chosen or alive
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
            # keep pin ages honest: session_affinity may short-circuit before the filter,
            # and an aging-out pin would let new chats pile onto a busy-but-idle lane again
            hit = self._pins.get(key)
            if hit is not None:
                self._pins[key] = (hit[0], time.monotonic())
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
            # load window for placement: the chosen lane gets an arrival (ground truth —
            # covers both the filter path and a session_affinity short-circuit)
            hu = _health_url({"litellm_params": {"api_base": line["url"]}})
            if hu:
                self._arrivals.setdefault(hu, deque(maxlen=512)).append(time.monotonic())
            d = os.path.dirname(REQLOG)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(REQLOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 - logging must never break the response path
            pass


handler = RigAffinity()
