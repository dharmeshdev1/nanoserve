"""HTTP front end. One background thread owns the engine; request threads only wait.

The engine and its cache are touched by a single thread, so no locking is needed around
scheduling. Request threads hand work over through a queue and block on a future.
"""

from __future__ import annotations

import itertools
import json
import queue
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from microllm.tokenizer import BPETokenizer

from nanoserve.engine import Engine, Request, RequestState

EOT = "<|endoftext|>"
MAX_BODY_BYTES = 1_000_000


class EngineLoop:
    """Runs ``Engine.step`` on a dedicated thread and resolves futures as requests finish."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._inbox: queue.Queue[tuple[Request, Future]] = queue.Queue()
        self._futures: dict[int, Future] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._ids = itertools.count()
        self.failed: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        error = RuntimeError("server is shutting down")
        for fut in self._futures.values():
            if not fut.done():
                fut.set_exception(error)
        while True:
            try:
                _, fut = self._inbox.get_nowait()
            except queue.Empty:
                break
            fut.set_exception(error)

    def submit(self, **fields) -> Future:
        """Queue a request and return a future for its ``RequestState``."""
        if self.failed is not None:
            raise RuntimeError(f"engine stopped after an error: {self.failed!r}")
        fut: Future = Future()
        self._inbox.put((Request(id=next(self._ids), **fields), fut))
        return fut

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_inbox()
                if self.engine.idle:
                    time.sleep(0.001)
                    continue
                for state in self.engine.step():
                    self._futures.pop(state.request.id).set_result(state)
        except BaseException as exc:  # a crashed engine must fail waiters, not hang them
            self.failed = exc
            for fut in self._futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError(f"engine error: {exc!r}"))
            raise

    def _drain_inbox(self) -> None:
        while True:
            try:
                request, fut = self._inbox.get_nowait()
            except queue.Empty:
                return
            try:
                self.engine.submit(request)
            except Exception as exc:  # a bad request must never take the engine thread down
                fut.set_exception(exc)
            else:
                self._futures[request.id] = fut


def render(state: RequestState, tok: BPETokenizer) -> dict:
    req = state.request
    out = state.output
    visible = out[:-1] if state.finish_reason == "eos" else out
    return {
        "text": tok.decode(visible),
        "tokens": out,
        "finish_reason": state.finish_reason,
        "usage": {"prompt_tokens": len(req.prompt), "completion_tokens": len(out)},
        "timing_ms": {
            "time_to_first_token": round((state.first_token_at - state.submitted_at) * 1000, 2),
            "total": round((state.finished_at - state.submitted_at) * 1000, 2),
        },
        "preemptions": state.preemptions,
    }


def _int(body: dict, key: str, default: int | None) -> int | None:
    value = body.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _float(body: dict, key: str, default: float | None) -> float | None:
    value = body.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def make_handler(loop: EngineLoop, tok: BPETokenizer, timeout_s: float):
    eos = tok.token_id(EOT) if EOT in tok.special_tokens else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # keep test and benchmark output quiet
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, close: bool = False) -> None:
            if close:
                # The request body was not read. On a keep-alive connection those bytes
                # would be parsed as the start of the next request, so end the connection.
                self.close_connection = True
            self._send(status, {"error": message})

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                if loop.failed is not None:
                    return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "engine stopped")
                return self._send(HTTPStatus.OK, {"status": "ok"})
            if self.path == "/metrics":
                e = loop.engine
                return self._send(HTTPStatus.OK, {
                    "waiting": len(e.waiting),
                    "running": len(e.running),
                    "free_blocks": e.cache.allocator.num_free,
                    "total_blocks": e.cache.allocator.num_blocks,
                    "steps": e.stats.steps,
                    "preemptions": e.stats.preemptions,
                })
            self._error(HTTPStatus.NOT_FOUND, "unknown path")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/completions":
                return self._error(HTTPStatus.NOT_FOUND, "unknown path", close=True)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error(HTTPStatus.BAD_REQUEST, "invalid Content-Length", close=True)
            if not 0 <= length <= MAX_BODY_BYTES:
                return self._error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body size is not allowed",
                    close=True,
                )
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                prompt = body.get("prompt", "")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                prompt_ids = tok.encode(prompt, allow_special=False)
                if not prompt_ids and eos is not None:
                    prompt_ids = [eos]  # an empty prompt starts a fresh document
                fields = dict(
                    prompt=prompt_ids,
                    max_new_tokens=_int(body, "max_tokens", 64),
                    temperature=_float(body, "temperature", 0.0),
                    top_k=_int(body, "top_k", None),
                    top_p=_float(body, "top_p", None),
                    eos_id=eos,
                    seed=_int(body, "seed", 0),
                )
                future = loop.submit(**fields)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except RuntimeError as exc:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            try:
                state = future.result(timeout=timeout_s)
            except FutureTimeout:
                return self._error(HTTPStatus.GATEWAY_TIMEOUT, "generation timed out")
            except (ValueError, TypeError) as exc:  # rejected by the engine at submission
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except RuntimeError as exc:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except Exception:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")
            self._send(HTTPStatus.OK, render(state, tok))

    return Handler


class _Server(ThreadingHTTPServer):
    # The default backlog of 5 resets connections when a burst of clients arrives at once.
    request_queue_size = 128
    daemon_threads = True


def create_server(
    engine: Engine, tok: BPETokenizer, host: str = "127.0.0.1", port: int = 8000,
    timeout_s: float = 120.0,
) -> tuple[ThreadingHTTPServer, EngineLoop]:
    loop = EngineLoop(engine)
    server = _Server((host, port), make_handler(loop, tok, timeout_s))
    loop.start()
    return server, loop
