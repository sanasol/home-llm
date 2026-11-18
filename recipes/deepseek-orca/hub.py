"""HF Hub helpers: range reads, safetensors/GGUF headers without downloading whole files."""
import json, os, re, struct, urllib.request, urllib.parse

_p = os.path.expanduser("~/.cache/huggingface/token")
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN and os.path.exists(_p):
    with open(_p) as _source:
        TOKEN = _source.read().strip()


def _origin(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only credential-free HTTPS URLs are allowed")
    return parsed.hostname.lower(), parsed.port or 443


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old_origin, new_origin = _origin(req.full_url), _origin(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and old_origin != new_origin:
            redirected.remove_header("Authorization")
        return redirected


_OPENER = urllib.request.build_opener(SafeRedirect())


def urlopen(request, timeout=120):
    _origin(request.full_url if isinstance(request, urllib.request.Request) else request)
    return _OPENER.open(request, timeout=timeout)


def _req(url, headers=None):
    origin = _origin(url)
    r = urllib.request.Request(url, headers=headers or {})
    if TOKEN and origin == ("huggingface.co", 443):
        r.add_header("Authorization", "Bearer " + TOKEN)
    return r


def rng(repo, path, start, end, rev="main"):
    """Bytes [start, end] inclusive; clip only at a server-declared file end."""
    if start < 0 or end < start:
        raise ValueError("Invalid HTTP byte range")
    for attempt in range(5):
        try:
            with urlopen(_req(f"https://huggingface.co/{repo}/resolve/{rev}/{path}",
                                             {"Range": f"bytes={start}-{end}"}), timeout=120) as f:
                expected = end - start + 1
                if f.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", f.headers.get("Content-Range", ""))
                    if not match:
                        raise ValueError("Invalid HTTP Content-Range")
                    first, last, total = map(int, match.groups())
                    if first != start or total <= first or last != min(end, total - 1):
                        raise ValueError("Invalid HTTP range response")
                    expected = last - first + 1
                elif f.status == 200 and start == 0:
                    length = f.headers.get("Content-Length")
                    if length is not None:
                        length = int(length)
                        if length < 0 or length > expected:
                            raise ValueError("Server ignored HTTP byte range")
                        expected = length
                else:
                    raise ValueError("Server ignored HTTP byte range")
                length = f.headers.get("Content-Length")
                if length is not None and int(length) != expected:
                    raise ValueError("HTTP Content-Length disagrees with byte range")
                data = f.read(expected + 1)
                if len(data) != expected:
                    raise EOFError(f"Incomplete HTTP range: expected {expected}, got {len(data)}")
                return data
        except Exception:
            if attempt == 4:
                raise


def api(path):
    with urlopen(_req("https://huggingface.co/api/" + path), timeout=60) as f:
        return json.load(f)


def st_header(repo, path, rev="main"):
    """(tensor dict without __metadata__, absolute offset of the data section, raw metadata)."""
    n = struct.unpack("<Q", rng(repo, path, 0, 7, rev))[0]
    h = json.loads(rng(repo, path, 8, 8 + n - 1, rev))
    meta = h.pop("__metadata__", None)
    return h, 8 + n, meta


def st_tensor(repo, path, name, hdr=None, rev="main"):
    """Raw bytes + info of one tensor."""
    h, off, _ = hdr or st_header(repo, path, rev)
    a, b = h[name]["data_offsets"]
    return rng(repo, path, off + a, off + b - 1, rev), h[name]


GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
        20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_header(buf):
    """Parse a GGUF header from bytes: ({name: (dims, type_id, offset)}, kv dict of scalars/strings, header end)."""
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, buf, pos)[0]
        pos += struct.calcsize("<" + fmt)
        return v

    def rs():
        nonlocal pos
        n = rd("Q")
        s = buf[pos:pos + n]
        if len(s) < n:
            raise EOFError
        pos += n
        return s.decode("utf-8", "replace")

    assert buf[:4] == b"GGUF"
    pos = 4
    rd("I")
    nt, nkv = rd("Q"), rd("Q")
    kv = {}
    fm = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    for _ in range(nkv):
        key, t = rs(), rd("I")
        if t == 8:
            kv[key] = rs()
        elif t == 9:
            it, n = rd("I"), rd("Q")
            if it == 8:
                for _ in range(n):
                    rs()
            else:
                pos += n * _SZ[it]
                if pos > len(buf):
                    raise EOFError
            kv[key] = f"<array {n}>"
        else:
            kv[key] = rd(fm[t])
    ts = {}
    for _ in range(nt):
        name = rs()
        nd = rd("I")
        dims = [rd("Q") for _ in range(nd)]
        ts[name] = (dims, rd("I"), rd("Q"))
    return ts, kv, pos
