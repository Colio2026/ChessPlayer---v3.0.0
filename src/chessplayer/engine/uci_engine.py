from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BestMove:
    uci:      str
    ponder:   str | None       = None
    score_cp: int | None       = None
    mate_in:  int | None       = None
    pv_uci:   list[str] | None = None


@dataclass
class PVLine:
    """One MultiPV line as it streams in."""
    rank:     int
    depth:    int            = 0
    score_cp: int | None     = None
    mate_in:  int | None     = None
    pv_uci:   list[str]      = field(default_factory=list)


class UciEngine:
    """
    UCI engine wrapper with streaming analysis support.

    Streaming API (Lichess-style):
        start_analysis(moves, multipv, threads, hash_mb, depth)
            → sends "go infinite" or "go depth N", returns immediately
        send_stop()
            → sends "stop", does NOT wait (bestmove arrives via readline)
        readline()
            → blocking read of one line from engine stdout
            → call from a dedicated reader thread

    Blocking API (kept for compatibility):
        analyze_movetime / analyze_multipv
    """

    def __init__(self, engine_exe: Path) -> None:
        self.engine_exe = engine_exe
        self.p: subprocess.Popen[str] | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

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

    def quit(self) -> None:
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

    # keep old name as alias
    def stop(self) -> None:
        self.quit()

    # ── streaming API ─────────────────────────────────────────────────────────

    def start_analysis(
        self,
        moves:    list[str],
        multipv:  int        = 1,
        threads:  int        = 1,
        hash_mb:  int        = 64,
        depth:    int | None = None,
    ) -> None:
        """
        Configure engine and start analysis. Returns immediately.
        Only sends setoption when values differ from last sent values.
        Does NOT send ucinewgame — that resets the hash which kills perf.
        """
        self.start()
        assert self.p is not None
        multipv = max(1, min(multipv, 10))
        # Only resend options if they changed
        if getattr(self, "_last_multipv", None) != multipv:
            self._cmd(f"setoption name MultiPV value {multipv}")
            self._last_multipv = multipv
        if getattr(self, "_last_threads", None) != threads:
            self._cmd(f"setoption name Threads value {max(1, threads)}")
            self._last_threads = threads
        if getattr(self, "_last_hash", None) != hash_mb:
            self._cmd(f"setoption name Hash value {max(1, hash_mb)}")
            self._last_hash = hash_mb
        self.set_position_uci(moves)
        if depth and depth > 0:
            self._cmd(f"go depth {int(depth)}")
        else:
            self._cmd("go infinite")

    def send_stop(self) -> None:
        """
        Send stop command. Does NOT wait for bestmove — that arrives
        via the reader thread calling readline().
        """
        try:
            self._cmd("stop")
        except Exception:
            pass

    def readline(self) -> str:
        """Blocking read of one line. Call from a dedicated reader thread."""
        if not self.p or not self.p.stdout:
            return ""
        try:
            return self.p.stdout.readline().strip()
        except Exception:
            return ""

    def set_position_uci(self, moves: list[str]) -> None:
        self._cmd(
            "position startpos"
            + ("" if not moves else " moves " + " ".join(moves))
        )

    # ── blocking API (kept for compatibility) ─────────────────────────────────

    def analyze_movetime(self, moves: list[str], movetime_ms: int) -> BestMove:
        results = self.analyze_multipv(moves, movetime_ms=movetime_ms, num_lines=1)
        return results[0] if results else BestMove(uci="0000")

    def analyze_multipv(
        self,
        moves:       list[str],
        movetime_ms: int        = 250,
        num_lines:   int        = 1,
        depth:       int | None = None,
        threads:     int | None = None,
        hash_mb:     int | None = None,
    ) -> list[BestMove]:
        self.start()
        assert self.p is not None
        num_lines = max(1, min(num_lines, 10))
        if threads is not None:
            self._cmd(f"setoption name Threads value {max(1, threads)}")
        if hash_mb is not None:
            self._cmd(f"setoption name Hash value {max(1, hash_mb)}")
        self._cmd(f"setoption name MultiPV value {num_lines}")
        self._cmd("ucinewgame")
        self._cmd("isready")
        self._wait_for("readyok", timeout_s=10.0)
        self.set_position_uci(moves)
        if depth and depth > 0:
            self._cmd(f"go depth {int(depth)}")
        else:
            self._cmd(f"go movetime {int(movetime_ms)}")

        best: dict[int, dict] = {}
        bestmove_uci = "0000"
        use_depth = bool(depth and depth > 0)
        t0        = time.time()
        timeout_s = 120.0 if use_depth else max(3.0, movetime_ms / 1000.0 + 5.0)

        while time.time() - t0 < timeout_s:
            line = self._readline()
            if not line:
                continue
            if line.startswith("info "):
                sc_cp, sc_mate, pv, idx = _parse_multipv_line(line)
                if idx is None:
                    idx = 1
                entry = best.setdefault(idx, {"score_cp": None, "mate_in": None, "pv": []})
                if sc_cp is not None:
                    entry["score_cp"] = sc_cp
                    entry["mate_in"]  = None
                if sc_mate is not None:
                    entry["mate_in"]  = sc_mate
                    entry["score_cp"] = None
                if pv:
                    entry["pv"] = pv
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove_uci = parts[1] if len(parts) >= 2 else "0000"
                break

        results: list[BestMove] = []
        for idx in sorted(best.keys()):
            e  = best[idx]
            pv = e["pv"] or []
            results.append(BestMove(
                uci=pv[0] if pv else (bestmove_uci if idx == 1 else "0000"),
                score_cp=e["score_cp"],
                mate_in=e["mate_in"],
                pv_uci=pv if pv else None,
            ))
        if not results:
            results.append(BestMove(uci=bestmove_uci))
        return results

    # ── internals ─────────────────────────────────────────────────────────────

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


# ── parsers ───────────────────────────────────────────────────────────────────

def _parse_multipv_line(
    line: str,
) -> tuple[int | None, int | None, list[str] | None, int | None]:
    score_cp: int | None       = None
    mate_in:  int | None       = None
    pv:       list[str] | None = None
    multipv:  int | None       = None
    depth:    int | None       = None

    parts = line.split()
    try:
        if "multipv" in parts:
            multipv = int(parts[parts.index("multipv") + 1])
        if "depth" in parts:
            depth = int(parts[parts.index("depth") + 1])
        if "score" in parts:
            i    = parts.index("score")
            kind = parts[i + 1]
            val  = parts[i + 2]
            if kind == "cp":
                score_cp = int(val)
            elif kind == "mate":
                mate_in = int(val)
        if "pv" in parts:
            j  = parts.index("pv")
            pv = parts[j + 1:]
    except Exception:
        pass

    return score_cp, mate_in, pv, multipv


def _parse_info_line(line: str) -> tuple[int | None, int | None, list[str] | None]:
    sc_cp, sc_mate, pv, _ = _parse_multipv_line(line)
    return sc_cp, sc_mate, pv
