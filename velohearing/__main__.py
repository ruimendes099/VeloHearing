import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .ingest import IngestError, ingest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="velohearing")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="normalize recordings into a case")
    pi.add_argument("--case", required=True, help="case id (letters, digits, . _ -)")
    pi.add_argument("files", nargs="+", type=Path)

    pt = sub.add_parser("transcribe", help="transcribe a case's ingested recordings")
    pt.add_argument("--case", required=True)
    pt.add_argument("--model", help="override transcribe.model from config")
    pt.add_argument("--language", help="override transcribe.language from config")
    pt.add_argument("--force", action="store_true", help="redo existing transcripts")

    pa = sub.add_parser("analyze", help="multi-agent analysis and imperceptibility report")
    pa.add_argument("--case", required=True)
    pa.add_argument("--no-review", action="store_true",
                    help="skip the text reviewer (nothing leaves the server)")
    pa.add_argument("--force", action="store_true", help="rerun agents even if cached")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "ingest":
        failed = 0
        for f in args.files:
            try:
                rec = ingest(f, args.case, cfg)
                print(f"{rec['id']}  {rec['duration_s']:>9.1f}s  {rec['source_name']}")
            except IngestError as e:
                print(f"error: {e}", file=sys.stderr)
                failed += 1
        return 1 if failed else 0

    if args.cmd == "transcribe":
        from dataclasses import replace

        from .transcribe import transcribe_case

        overrides = {k: v for k, v in (("model", args.model), ("language", args.language)) if v}
        if overrides:
            cfg = replace(cfg, transcribe=replace(cfg.transcribe, **overrides))
        try:
            transcribe_case(args.case, cfg, force=args.force)
        except IngestError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "analyze":
        from .analyze import analyze_case
        from .review import ReviewUnavailable

        try:
            analyze_case(args.case, cfg, review=not args.no_review, force=args.force)
        except (IngestError, ReviewUnavailable) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            import anthropic

            if not isinstance(e, anthropic.APIError):
                raise
            # agent outputs are cached, so rerunning only repeats the review
            print(f"error: Claude API failed during review: {e}", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
