from __future__ import annotations

import argparse
from pathlib import Path

from microllm.data import load_tokens
from microllm.runs import load_run

from nanoserve.bench import benchmark, format_table, make_workload
from nanoserve.engine import Engine
from nanoserve.server import create_server


def cmd_serve(a: argparse.Namespace) -> None:
    model, tok, _ = load_run(a.run, a.device)
    engine = Engine(model, num_blocks=a.num_blocks, block_size=a.block_size, max_batch=a.max_batch)
    server, loop = create_server(engine, tok, a.host, a.port)
    print(f"serving {a.run} on http://{a.host}:{a.port}  (POST /v1/completions)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        loop.stop()


def cmd_bench(a: argparse.Namespace) -> None:
    model, _, cfg = load_run(a.run, a.device)
    tokens = load_tokens(Path(a.data or cfg["data_dir"]) / "val.bin")
    block = model.cfg.block_size
    reqs = make_workload(
        tokens, a.requests, (a.min_prompt, a.max_prompt), (a.min_new, a.max_new), block, a.seed
    )
    results = benchmark(model, reqs, a.batch, a.num_blocks, a.block_size)
    print(f"{len(reqs)} requests, all submitted at once; greedy outputs verified identical\n")
    print(format_table(results, contiguous_tokens=max(a.batch) * block))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nanoserve", description="Small LLM inference engine")
    sub = p.add_subparsers(dest="command", required=True)

    def common(s: argparse.ArgumentParser) -> None:
        s.add_argument("--run", required=True, help="a microllm run directory")
        s.add_argument("--device", default="auto")
        s.add_argument("--num-blocks", type=int, default=512)
        s.add_argument("--block-size", type=int, default=16)

    s = sub.add_parser("serve", help="start the HTTP server")
    common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-batch", type=int, default=32)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("bench", help="batching versus sequential serving")
    common(s)
    s.add_argument("--data")
    s.add_argument("--requests", type=int, default=48)
    s.add_argument("--min-prompt", type=int, default=8)
    s.add_argument("--max-prompt", type=int, default=48)
    s.add_argument("--min-new", type=int, default=16)
    s.add_argument("--max-new", type=int, default=128)
    s.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        parser.exit(2, f"nanoserve: error: {exc}\n")


if __name__ == "__main__":
    main()
