"""Offline policy test for rig_affinity placement (run with /opt/litellm/venv/bin/python3)."""
import importlib.util, sys, time

spec = importlib.util.spec_from_file_location("rig_affinity", "~/rig_affinity.new.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

U3, U4 = "http://127.0.0.1:8003/health", "http://127.0.0.1:8004/health"
URLS = [U3, U4]
fails = []

def ck(name, cond, got=None):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  got={got!r}"))
    if not cond:
        fails.append(name)

def fresh():
    ra.PIN_TTL, ra.ARRIVAL_WINDOW, ra.ACTIVE_WINDOW = 3600.0, 300.0, 180.0
    ra.MOVE_COOLDOWN, ra.MIN_MOVE_GAP = 600.0, 30.0
    return ra.RigAffinity()

def arrive(h, url, n=1, age=0.0):
    dq = h._arrivals.setdefault(url, __import__("collections").deque(maxlen=512))
    for _ in range(n):
        dq.append(time.monotonic() - age)

# 1) idle fresh keys alternate (tie -> RR)
h = fresh()
a, b = h._place("kA", URLS), h._place("kB", URLS)
ck("idle fan-out spreads", {a, b} == {U3, U4}, (a, b))

# 2) stickiness: follow-up turn stays on its lane even when the other looks idle
lane = h._place("kA", URLS)
ck("stickiness", lane == a, lane)

# 3) ghost-pin replay (the 06.11 incident): two one-shot aux calls pinned 8003,
#    a chatty agent streams on 8004; a NEW agent key must go to 8003 (score-based)
h = fresh()
h._place("one1", URLS)                                # one-shots: pin + score forced below
h._pins["one1"] = (U3, time.monotonic()); arrive(h, U3)
h._pins["one2"] = (U3, time.monotonic()); arrive(h, U3)
h._pins["agent1"] = (U4, time.monotonic()); arrive(h, U4, n=8)
got = h._place("agent2", URLS)
ck("new agent avoids the busy lane despite fewer pins", got == U3, got)

# 4) congestion rebalance: two active pins on 8004, none on 8003 -> the NEWEST moves on its turn
h = fresh()
now = time.monotonic()
h._pins["old"] = (U4, now - 10)     # older active conversation
h._pins["new"] = (U4, now - 1)      # newest active conversation
arrive(h, U4, n=6); arrive(h, U3, n=2)
got = h._place("new", URLS)
ck("rebalance moves the newest pin", got == U3, got)
got = h._place("old", URLS)
ck("rebalance leaves the older pin (not newest)", got == U4, got)

# 5) guards: the moved key must not move again immediately (per-key cooldown)...
got = h._place("new", URLS)
ck("move cooldown holds", got == U3, got)
#    ...and MIN_MOVE_GAP blocks a second move right after
h._pins["third"] = (U4, time.monotonic())
got = h._place("third", URLS)
ck("converged diff=1: no stampede move", got == U4, got)

# 6) diff==1 must NOT trigger a move (converged state, no churn)
h = fresh()
now = time.monotonic()
h._pins["x"] = (U4, now - 1); h._pins["y"] = (U4, now - 2); h._pins["z"] = (U3, now - 3)
arrive(h, U4, n=5); arrive(h, U3, n=5)
got = h._place("y", URLS)
ck("no move when diff is 1", got == U4, got)

# 7) dead pinned lane re-places onto a live one
h = fresh()
h._pins["k"] = ("http://127.0.0.1:9999/health", time.monotonic())
got = h._place("k", URLS)
ck("dead lane re-place", got in URLS, got)

# 8) window decay: old arrivals stop counting; stale pins stop counting as active
h = fresh()
arrive(h, U4, n=9, age=290.0)
h._pins["ghost"] = (U4, time.monotonic() - 200.0)   # beyond ACTIVE_WINDOW
h._pins["live"] = (U3, time.monotonic() - 5.0)
got = h._place("fresh1", URLS)
ck("aged arrivals still count within ARRIVAL_WINDOW", got == U3, got)
h._pins["ghost"] = (U4, time.monotonic() - 400.0)
h._arrivals[U4].clear()
got = h._place("fresh2", URLS)
ck("idle lane wins after decay", got in URLS, got)

# 9) prune: PIN_TTL expiry removes pins (shrink TTL and rewind state)
h = fresh()
ra.PIN_TTL = 10.0
h._pins["gone"] = (U4, time.monotonic() - 30.0)
h._place("probe", URLS)
ck("PIN_TTL prune", "gone" not in h._pins)

print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
