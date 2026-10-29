import http.client
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from microllm.generate import generate
from microllm.model import GPT, ModelConfig
from microllm.tokenizer import BPETokenizer

from nanoserve.bench import percentile
from nanoserve.engine import Engine
from nanoserve.server import create_server

EOT = "<|endoftext|>"
TEXT = "the cat sat on the mat and the dog ran in the park. once upon a time there was a bird. " * 8


@pytest.fixture(scope="module")
def stack():
    tok = BPETokenizer.train(TEXT, 300, [EOT])
    torch.manual_seed(0)
    model = GPT(
        ModelConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=4,
                    n_kv_head=2, n_embd=32)
    ).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    engine = Engine(model, num_blocks=64, block_size=4, max_batch=8)
    server, loop = create_server(engine, tok, port=0, timeout_s=30)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", model, tok, loop
    server.shutdown()
    server.server_close()
    loop.stop()


def call(base, path, payload=None, raw=None, method="POST"):
    if raw is not None:
        data = raw
    else:
        data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_health_and_metrics(stack):
    base, *_ = stack
    assert call(base, "/health", method="GET") == (200, {"status": "ok"})
    status, metrics = call(base, "/metrics", method="GET")
    assert status == 200 and metrics["total_blocks"] == 64


def test_completion_matches_direct_greedy_generation(stack):
    base, model, tok, _ = stack
    prompt = "the cat"
    status, body = call(base, "/v1/completions", {"prompt": prompt, "max_tokens": 12})
    want = generate(model, tok.encode(prompt, allow_special=False), 12, temperature=0,
                    eos_id=tok.token_id(EOT))
    assert status == 200 and body["tokens"] == want
    assert body["usage"]["completion_tokens"] == len(want)
    assert body["timing_ms"]["total"] >= body["timing_ms"]["time_to_first_token"] >= 0


def test_concurrent_requests_all_get_their_own_correct_answer(stack):
    base, model, tok, _ = stack
    prompts = [f"the {w}" for w in ("cat", "dog", "bird", "mat", "park", "time")] * 3

    def one(prompt):
        return prompt, call(base, "/v1/completions", {"prompt": prompt, "max_tokens": 10})

    with ThreadPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(one, prompts))
    for prompt, (status, body) in results:
        want = generate(model, tok.encode(prompt, allow_special=False), 10, temperature=0,
                        eos_id=tok.token_id(EOT))
        assert status == 200 and body["tokens"] == want, prompt


def test_empty_prompt_is_allowed(stack):
    base, *_ = stack
    status, body = call(base, "/v1/completions", {"prompt": "", "max_tokens": 4})
    assert status == 200 and body["usage"]["completion_tokens"] >= 1


def test_prompt_cannot_inject_the_end_of_text_token(stack):
    base, _, tok, _ = stack
    status, body = call(base, "/v1/completions", {"prompt": f"a{EOT}b", "max_tokens": 2})
    assert status == 200 and body["usage"]["prompt_tokens"] == len(
        tok.encode(f"a{EOT}b", allow_special=False)
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": 5},
        {"prompt": "x", "max_tokens": "many"},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "max_tokens": 10**9},
        {"prompt": "x", "temperature": "hot"},
        {"prompt": "x", "temperature": -1},
        {"prompt": "x", "top_k": "x"},
        {"prompt": "x", "top_k": 0},
        {"prompt": "x", "top_k": 1.5},
        {"prompt": "x", "top_p": 7},
        {"prompt": "x", "seed": True},
        [1, 2, 3],
    ],
)
def test_bad_input_is_a_400_and_never_takes_the_engine_down(stack, payload):
    base, _, _, loop = stack
    status, body = call(base, "/v1/completions", payload)
    assert status == 400 and "error" in body
    assert loop.failed is None
    assert call(base, "/health", method="GET")[0] == 200


def test_malformed_json_and_unknown_paths(stack):
    base, *_ = stack
    assert call(base, "/v1/completions", raw=b"{not json")[0] == 400
    assert call(base, "/nope", {"a": 1})[0] == 404
    assert call(base, "/nope", method="GET")[0] == 404


@pytest.mark.parametrize("claimed", ["2000000", "-5"])
def test_unacceptable_body_sizes_are_rejected_and_the_connection_is_closed(stack, claimed):
    base, *_ = stack
    conn = http.client.HTTPConnection(base.removeprefix("http://"), timeout=10)
    conn.putrequest("POST", "/v1/completions")
    conn.putheader("Content-Length", claimed)  # claim a size without sending that many bytes
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413 and resp.getheader("Connection") == "close"
    conn.close()


def test_rejecting_an_unread_body_does_not_corrupt_later_requests(stack):
    base, *_ = stack
    conn = http.client.HTTPConnection(base.removeprefix("http://"), timeout=10)
    conn.request("POST", "/wrong", body=b'{"prompt": "x"}')
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 404 and resp.getheader("Connection") == "close"
    conn.close()
    assert call(base, "/v1/completions", {"prompt": "the", "max_tokens": 2})[0] == 200


def test_a_burst_of_simultaneous_connections_is_all_accepted(stack):
    base, *_ = stack
    with ThreadPoolExecutor(max_workers=40) as pool:
        codes = list(pool.map(lambda _: call(base, "/health", method="GET")[0], range(120)))
    assert codes == [200] * 120


def test_sampling_is_reproducible_for_a_given_seed(stack):
    base, *_ = stack
    req = {"prompt": "the", "max_tokens": 10, "temperature": 1.0, "top_k": 20, "seed": 3}
    first = call(base, "/v1/completions", req)[1]["tokens"]
    assert first == call(base, "/v1/completions", req)[1]["tokens"]


def test_blocks_are_all_free_after_traffic(stack):
    base, *_ = stack
    status, metrics = call(base, "/metrics", method="GET")
    assert metrics["running"] == 0 and metrics["free_blocks"] == metrics["total_blocks"]


def test_percentile_nearest_rank():
    data = [5, 1, 4, 2, 3]
    assert [percentile(data, q) for q in (0, 20, 50, 80, 100)] == [1, 1, 3, 4, 5]
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)
