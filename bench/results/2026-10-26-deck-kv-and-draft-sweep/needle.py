"""Needle in a haystack at a long prompt: a unique fact placed at several depths of source-code filler.
argv: URL MODEL OUT.json TARGET_TOKENS [SRC_DIR ...]. Greedy, thinking off; passes when the reply holds the code."""
import glob, json, os, random, sys, time, urllib.request

url, model, out, target = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], int(sys.argv[4])
dirs = sys.argv[5:] or [os.path.expanduser("~/ninfer/src")]
filler = ""
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "**/*.*"), recursive=True)):
        if f.endswith((".cpp", ".h", ".hpp", ".cu", ".cuh", ".zig", ".py")):
            filler += f"\n// ==== {os.path.basename(f)} ====\n" + open(f, errors="ignore").read()
chars = int(target * 3.2)  # code tokenizes at ~3.2 chars/token for Qwen
if len(filler) < chars:
    sys.exit(f"filler has {len(filler)} chars, need {chars}")
filler = filler[:chars]

res = []
for depth in (0.1, 0.5, 0.9):
    code = f"{random.choice('ABCDEFGHJK')}{random.randint(1000, 9999)}-{random.choice('MNPQRSTUVW')}{random.randint(10, 99)}"
    needle = f"\n// NOTE: the deployment passphrase for the deck cluster is {code}. Remember it.\n"
    cut = filler.rfind("\n", 0, int(len(filler) * depth)) + 1
    prompt = filler[:cut] + needle + filler[cut:] + \
        "\n\nWhat is the deployment passphrase for the deck cluster mentioned in a NOTE comment above? Reply with the passphrase only."
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    answer = r["choices"][0]["message"]["content"] or ""
    row = {"depth": depth, "prompt_tokens": r["usage"]["prompt_tokens"], "expected": code, "answer": answer.strip(),
           "pass": code in answer, "seconds": round(time.time() - t, 1)}
    res.append(row)
    print(f"needle depth {depth}: {'PASS' if row['pass'] else 'FAIL'} ({row['prompt_tokens']} tok, {row['seconds']} s) "
          f"expected {code} got {answer.strip()[:40]!r}", flush=True)
json.dump(res, open(out, "w"), indent=1)
