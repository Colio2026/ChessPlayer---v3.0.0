import argparse

from config.loader import load_config
from app import run_app
from pgn.indexer import build_or_update_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chessplayer")
    parser.add_argument("--index", action="store_true", help="Build/update PGN index and exit.")
    args = parser.parse_args(argv)

    config = load_config()

    if args.index:
        build_or_update_index(config, progress_cb=None, cancel_cb=None)
        return 0

    run_app(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
