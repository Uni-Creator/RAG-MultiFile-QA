"""python -m rag.cli {ingest|ask|eval|stats} ...   (add --offline to run with no models / API key)"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import RAGConfig
from .evaluation import compare_reports, evaluate, load_dataset
from .pipeline import RAGPipeline
from .types import Principal


def build(args) -> tuple:
    cfg = RAGConfig.from_env()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    kw = {}
    if args.offline:
        from .embeddings import HashEmbedder
        from .testing import FakeLLM
        cfg.use_reranker = cfg.use_nli = False
        cfg.min_dense_sim = -1.0   # hash embedder scores are not semantic; FakeLLM does the abstaining
        kw = {"llm": FakeLLM(), "embedder": HashEmbedder()}
    return RAGPipeline(cfg, **kw), Principal(args.tenant, args.user, tuple(args.roles.split(",")))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--user", default="local")
    ap.add_argument("--roles", default="member")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("eval"); p.add_argument("dataset"); p.add_argument("--name", default="run")
    p.add_argument("--baseline"); p.add_argument("--no-generate", action="store_true")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    pipe, who = build(args)

    if args.cmd == "ingest":
        files = [(os.path.basename(f), open(f, "rb").read()) for f in args.paths]
        print(json.dumps(pipe.ingest(who, files), indent=2))
    elif args.cmd == "ask":
        for ev in pipe.ask_stream(who, args.question):
            if ev["type"] == "token" and not args.json:
                sys.stdout.write(ev["text"]); sys.stdout.flush()
            elif ev["type"] == "final":
                a = ev["answer"]
                print(json.dumps(a.to_dict(), indent=2, default=str) if args.json else
                      f"\n\n[groundedness={a.groundedness} confidence={a.confidence} abstained={a.abstained}]\n"
                      + "\n".join(f"  [{c['label']}] {c['filename']} p.{c['page']} — {c['section']}" for c in a.citations))
    elif args.cmd == "eval":
        rep = evaluate(pipe, who, load_dataset(args.dataset), generate=not args.no_generate, name=args.name)
        print(json.dumps(rep["metrics"], indent=2))
        if args.baseline:
            reg = compare_reports(json.load(open(args.baseline)), rep)
            print("REGRESSIONS:" if reg else "no regressions", *(json.dumps(r) for r in reg), sep="\n")
            sys.exit(1 if reg else 0)
    elif args.cmd == "stats":
        print(json.dumps(pipe.stats(who), indent=2, default=str))


if __name__ == "__main__":
    main()
