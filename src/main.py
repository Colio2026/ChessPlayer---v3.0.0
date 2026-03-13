import argparse
from pathlib import Path

from chessplayer.config.loader import load_config
from chessplayer.pgn.indexer import (
    build_or_rebuild_index_for_source,
    build_or_rebuild_index_for_sources,
    build_or_update_index,
)
from app import run_app



def _detect_source_type(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if path.is_dir():
        return "directory"
    return "archive_file"


def _parse_cli_sources(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        source_type = _detect_source_type(value)
        parsed.append((source_type, value))
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chessplayer")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Build/update the default configured PGN index and exit.",
    )
    parser.add_argument(
        "--index-source",
        action="append",
        default=[],
        metavar="PATH",
        help="Rebuild and add a PGN file or PGN directory source to the index. Repeat for multiple sources.",
    )
    args = parser.parse_args(argv)

    config = load_config()

    if args.index_source:
        sources = _parse_cli_sources(args.index_source)
        if len(sources) == 1:
            source_type, source_path = sources[0]
            build_or_rebuild_index_for_source(
                config,
                source_type=source_type,
                source_path=source_path,
                progress_cb=None,
                cancel_cb=None,
            )
        else:
            build_or_rebuild_index_for_sources(
                config,
                sources=sources,
                progress_cb=None,
                cancel_cb=None,
            )
        return 0

    if args.index:
        build_or_update_index(config, progress_cb=None, cancel_cb=None)
        return 0

    run_app(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
