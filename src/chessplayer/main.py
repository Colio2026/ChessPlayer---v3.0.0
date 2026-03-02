import sys
from pathlib import Path
from chessplayer.pgn.indexer import build_index
from chessplayer.utils.paths import resolve_path

def run_ui():
    from chessplayer.app import run
    run()

def run_index():
    import yaml
    config_path = resolve_path("config/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pgn_path = resolve_path(config["pgn_sources"]["active_source"]["path"])
    db_path = resolve_path("data/index.sqlite")
    build_index(pgn_path, db_path)
    print("Index build complete.")

if __name__ == "__main__":
    if "--index" in sys.argv:
        run_index()
    else:
        run_ui()
