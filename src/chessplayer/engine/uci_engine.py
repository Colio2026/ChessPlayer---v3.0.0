from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BestMove:
    uci: str
    ponder: str | None = None
    score_cp: int | None = None          # centipawns from side-to-move perspective
    mate_in: int | None = None           # mate distance (+/-)
    pv_uci: list[str] | None = None      # principal variation (uci moves), best available


class UciEngine:
    """
    Minimal UCI engine wrapper sufficient for:
      - start/stop
      - set position
      - go movetime
      - capture latest evaluation + PV while thinking
    """

    def __init__(self, engine_exe: Path) -> None:
        self.engine_exe = engine_exe
        self.p: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.p is not None:
            return
        self.p = subprocess.Popen(
            [str(self.engine_exe)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._cmd("uci")
        self._wait_for("uciok", timeout_s=10.0)
        self._cmd("isready")
        self._wait_for("readyok", timeout_s=10.0)

    def stop(self) -> None:
        if not self.p:
            return
        try:
            self._cmd("quit")
        except Exception:
            pass
        try:
            self.p.kill()
        except Exception:
            pass
        self.p = None

    def set_position_uci(self, moves: list[str]) -> None:
        self._cmd("position startpos" + ("" if not moves else " moves " + " ".join(moves)))

    def analyze_movetime(self, moves: list[str], movetime_ms: int) -> BestMove:
        """
        Runs `go movetime` and returns bestmove plus last seen score+pv while thinking.
        """
        self.start()
        assert self.p is not None

        # new game boundary
        self._cmd("ucinewgame")
        self._cmd("isready")
        self._wait_for("readyok", timeout_s=10.0)

        self.set_position_uci(moves)
        self._cmd(f"go movetime {int(movetime_ms)}")

        last_score_cp: int | None = None
        last_mate_in: int | None = None
        last_pv: list[str] | None = None

        t0 = time.time()
        timeout_s = max(3.0, movetime_ms / 1000.0 + 5.0)

        while time.time() - t0 < timeout_s:
            line = self._readline()
            if not line:
                continue

            if line.startswith("info "):
                sc_cp, sc_mate, pv = _parse_info_line(line)
                if sc_cp is not None:
                    last_score_cp = sc_cp
                    last_mate_in = None
                if sc_mate is not None:
                    last_mate_in = sc_mate
                    last_score_cp = None
                if pv:
                    last_pv = pv

            if line.startswith("bestmove"):
                parts = line.split()
                uci = parts[1] if len(parts) >= 2 else "0000"
                ponder = None
                if "ponder" in parts:
                    i = parts.index("ponder")
                    if i + 1 < len(parts):
                        ponder = parts[i + 1]
                return BestMove(
                    uci=uci,
                    ponder=ponder,
                    score_cp=last_score_cp,
                    mate_in=last_mate_in,
                    pv_uci=last_pv,
                )

        raise TimeoutError("Timed out waiting for bestmove")

    # -------- internals --------

    def _cmd(self, s: str) -> None:
        if not self.p or not self.p.stdin:
            raise RuntimeError("Engine not started")
        self.p.stdin.write(s.strip() + "\n")
        self.p.stdin.flush()

    def _readline(self) -> str:
        assert self.p is not None and self.p.stdout is not None
        return self.p.stdout.readline().strip()

    def _wait_for(self, token: str, timeout_s: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            line = self._readline()
            if token in line:
                return
        raise TimeoutError(f"Timed out waiting for {token}")


def _parse_info_line(line: str) -> tuple[int | None, int | None, list[str] | None]:
    """
    Parse UCI 'info' lines. We care about:
      - score cp <n>
      - score mate <n>
      - pv <moves...>
    Returns (score_cp, mate_in, pv_uci)
    """
    score_cp: int | None = None
    mate_in: int | None = None
    pv: list[str] | None = None

    parts = line.split()
    # Example:
    # info depth 18 seldepth 28 score cp 34 pv e2e4 e7e5 g1f3 ...
    # info depth 20 score mate 3 pv ...
    try:
        if "score" in parts:
            i = parts.index("score")
            if i + 2 < len(parts):
                kind = parts[i + 1]
                val = parts[i + 2]
                if kind == "cp":
                    score_cp = int(val)
                elif kind == "mate":
                    mate_in = int(val)
    except Exception:
        pass

    if "pv" in parts:
        j = parts.index("pv")
        if j + 1 < len(parts):
            pv = parts[j + 1 :]

    return score_cp, mate_in, pv