"""The league page's public API: patches between page states, and the push stream (server-sent events).

    GET /api/state.json   the whole state (league_page.api_state; serve.py writes the file, Caddy serves it)
    GET /api/events       this module's stream, behind Caddy: `patch` {from, v, ts, ops} after every change, `state` (the
                          whole state) when a browser connects behind, `ping` every PING seconds. Event ids are state
                          versions, so a reconnecting browser (Last-Event-ID) gets the whole state only if it missed a change.

Patch ops, applied in order (web/app.js applyPatch is the same):
    ["set", path, value]   ["del", path]   ["push", path, items]   ["order", path, keys]
A path is a list of dict keys, list indices, or "@<key>" for the element of a list of dicts with that "key" (matchups,
which re-sort as the odds move).
"""
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PING, MAX_CLIENTS, QUEUE = 15, 300, 64
STAMPS = ("v", "ts")


def _keyed(xs):
    return bool(xs) and all(isinstance(x, dict) and "key" in x for x in xs) and len({str(x["key"]) for x in xs}) == len(xs)


def diff(a, b, path=()):
    """Ops that turn a into b (both plain JSON)."""
    if type(a) is not type(b) or not isinstance(a, (dict, list)):
        return [] if type(a) is type(b) and a == b else [["set", list(path), b]]
    if isinstance(a, dict):
        ops = [["del", [*path, k]] for k in a if k not in b]
        for k, v in b.items():
            ops += diff(a[k], v, (*path, k)) if k in a else [["set", [*path, k], v]]
        return ops
    if _keyed(a) and _keyed(b):
        ka, kb = [str(x["key"]) for x in a], [str(x["key"]) for x in b]
        if set(ka) != set(kb):
            return [["set", list(path), b]]
        old = {str(x["key"]): x for x in a}
        ops = [op for x in b for op in diff(old[str(x["key"])], x, (*path, "@" + str(x["key"])))]
        if ka != kb:
            ops.append(["order", list(path), [x["key"] for x in b]])
        return ops
    if len(b) < len(a):
        return [["set", list(path), b]]
    ops = [op for i in range(len(a)) for op in diff(a[i], b[i], (*path, i))]
    if len(b) > len(a):
        ops.append(["push", list(path), b[len(a):]])
    return ops


def _locate(node, step):
    if isinstance(step, str) and step.startswith("@") and isinstance(node, list):
        return next(i for i, x in enumerate(node) if isinstance(x, dict) and str(x.get("key")) == step[1:])
    return step


def apply(state, ops):
    """The state after ops (a new object)."""
    out = json.loads(json.dumps(state))
    for op in ops:
        kind, path, value = op[0], op[1], (op[2] if len(op) > 2 else None)
        if not path:
            out = json.loads(json.dumps(value))
            continue
        node = out
        for step in path[:-1]:
            node = node[_locate(node, step)]
        k = _locate(node, path[-1])
        if kind == "set":
            node[k] = value
        elif kind == "del":
            del node[k]
        elif kind == "push":
            node[k].extend(value)
        elif kind == "order":
            by_key = {x["key"]: x for x in node[k]}
            node[k] = [by_key[x] for x in value]
    return out


def strip(state):
    return {k: v for k, v in state.items() if k not in STAMPS}


class Hub:
    """Connected browsers and the latest state. publish() sends each a patch; a browser that joins behind gets the whole
    state. A browser too slow to keep up (QUEUE events behind) is dropped; its EventSource reconnects."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.state, self.state_json = None, None

    def publish(self, state):
        prev = self.state
        data = json.dumps(state, separators=(",", ":"))
        if prev is None:
            msg = ("state", data, state["v"])
        else:
            msg = ("patch", json.dumps({"from": prev["v"], "v": state["v"], "ts": state["ts"],
                                        "ops": diff(strip(prev), strip(state))}, separators=(",", ":")), state["v"])
        with self.lock:
            self.state, self.state_json = state, data
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    self.clients.discard(q)
                    q.queue.clear()
                    q.put_nowait(None)
        return msg

    def join(self):
        with self.lock:
            if len(self.clients) >= MAX_CLIENTS:
                return None, None
            q = queue.Queue(QUEUE)
            self.clients.add(q)
            return q, (self.state_json, self.state["v"] if self.state else None)

    def leave(self, q):
        with self.lock:
            self.clients.discard(q)

    def count(self):
        with self.lock:
            return len(self.clients)


TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json"}


def handler(hub, static=None):
    root = Path(static).resolve() if static else None

    class Events(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send(self, event, data, eid=None):
            self.wfile.write((f"event: {event}\n" + (f"id: {eid}\n" if eid else "") + f"data: {data}\n\n").encode())
            self.wfile.flush()

        def file(self, path):
            """Local preview only (Caddy serves the files in production): a file under the static folder."""
            f = (root / (unquote(path).lstrip("/") or "index.html")).resolve()
            if root not in f.parents or not f.is_file():
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", TYPES.get(f.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path != "/api/events":
                self.file(u.path) if root else self.send_error(404)
                return
            last = self.headers.get("Last-Event-ID") or (parse_qs(u.query).get("v") or [""])[0]
            q, (snap, v) = hub.join()
            if q is None:
                self.send_error(503, "too many listeners")
                return
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.write(b"retry: 5000\n\n")
                if v is not None and last != v:
                    self.send("state", snap, v)
                while True:
                    try:
                        ev = q.get(timeout=PING)
                    except queue.Empty:
                        ev = ("ping", "{}", None)
                    if ev is None:
                        break
                    self.send(*ev)
            except OSError:
                pass
            finally:
                hub.leave(q)

        def log_message(self, *args):
            pass
    return Events


def start(hub, host="127.0.0.1", port=8787, static=None):
    """Serve /api/events on host:port in a background thread (and, for a local preview, the files under `static`);
    returns the server (server.server_address has the port)."""
    srv = ThreadingHTTPServer((host, port), handler(hub, static))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="api-events").start()
    return srv
