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

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    failed = 0
    for f in args.files:
        try:
            rec = ingest(f, args.case, cfg)
            print(f"{rec['id']}  {rec['duration_s']:>9.1f}s  {rec['source_name']}")
        except IngestError as e:
            print(f"error: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
