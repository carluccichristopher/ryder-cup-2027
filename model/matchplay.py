"""Hole-by-hole match play Monte Carlo engine.

Every player is reduced to a per-hole scoring distribution derived from their
form rating (strokes per round vs an average PGA Tour field). Matches are then
simulated hole by hole, thousands at a time, which produces:
  - P(win) / P(extra holes) / P(loss)
  - the EXACT final scoreline distribution: 10&8, 9&8 ... 2&1, 1UP, 19th, 20th...

Supports singles, four-ball (best ball of two) and foursomes (alternate shot).
"""
from __future__ import annotations

import numpy as np

from . import config

# score categories: eagle(-2), birdie(-1), par(0), bogey(+1), double_or_worse(+2)
LABELS = ("eagle", "birdie", "par", "bogey", "dbl")


def rating_to_lambda(rating: float) -> float:
    """Convert a per-round rating (strokes vs field) to the distribution shift."""
    return -rating / (18.0 * config.LAMBDA_SENSITIVITY)


def hole_dist(rating: float) -> dict:
    """Per-hole scoring distribution for a player with the given rating."""
    b = config.BASE_HOLE_DIST
    lam = float(np.clip(rating_to_lambda(rating), -0.18, 0.14))
    eagle = float(np.clip(b["eagle"] + 0.30 * lam, 0.0, 0.09))
    birdie = float(np.clip(b["birdie"] + 1.00 * lam, 0.02, 0.46))
    bogey = float(np.clip(b["bogey"] - 0.55 * lam, 0.015, 0.34))
    dbl = float(np.clip(b["dbl"] - 0.15 * lam, 0.003, 0.14))
    par = max(1.0 - eagle - birdie - bogey - dbl, 0.05)
    dist = np.array([eagle, birdie, par, bogey, dbl])
    dist = dist / dist.sum()
    return {lab: float(p) for lab, p in zip(LABELS, dist)}


def sim_dist(rating: float) -> dict:
    """Distribution used inside match sims: sharpens quality gaps so match-play
    win probabilities match real-world dynamics."""
    return hole_dist(rating * config.MATCH_SHARPNESS)


def sim_foursomes_dist(rating_a: float, rating_b: float) -> dict:
    return hole_dist((rating_a + rating_b) / 2.0 * config.MATCH_SHARPNESS
                     + config.FOURSOMES_PENALTY)


def dist_array(dist) -> np.ndarray:
    if isinstance(dist, dict):
        return np.array([dist[lab] for lab in LABELS], dtype=float)
    return np.asarray(dist, dtype=float)


def foursomes_dist(rating_a: float, rating_b: float) -> dict:
    """Alternate shot: partners' blended ability plus a rhythm penalty."""
    return hole_dist((rating_a + rating_b) / 2.0 + config.FOURSOMES_PENALTY)


def draw_scores(dist, rng, n, holes):
    """Draw (n, holes) categorical scores with values -2..+2."""
    probs = np.cumsum(dist_array(dist) / dist_array(dist).sum())
    u = rng.random((n, holes))
    return np.searchsorted(probs, u, side="right") - 2.0


# ---------------------------------------------------------------------------
# Core match resolution
# ---------------------------------------------------------------------------

def _decide(running, n_holes):
    """Where does each match end? running: (n, played) A-lead in holes won."""
    played = running.shape[1]
    remaining = n_holes - np.arange(1, played + 1)
    decided = np.abs(running) > remaining
    return decided, remaining


def _resolve_extra(a, b, rng, n_tied, max_extra=None):
    """Sudden-death holes for all-square matches.

    Returns (winners, holes): bool array (True = A wins) and the deciding
    hole number (19.., or -1 for still-tied after max_extra, ~impossible)."""
    max_extra = max_extra or config.MAX_EXTRA_HOLES
    winners = np.zeros(n_tied, dtype=bool)
    holes = np.full(n_tied, -1, dtype=int)
    pending = np.arange(n_tied)
    for h in range(1, max_extra + 1):
        if pending.size == 0:
            break
        sa = draw_scores(a, rng, pending.size, 1).ravel()
        sb = draw_scores(b, rng, pending.size, 1).ravel()
        d = np.sign(sb - sa)
        done = d != 0
        idx = pending[done]
        winners[idx[d[done] > 0]] = True
        holes[idx] = 18 + h
        pending = pending[~done]
    return winners, holes


def _summarise(diff_running, decided, remaining, extra_win, extra_hole, n, want_scorelines):
    """Turn per-match end states into probabilities (and scorelines if asked)."""
    ended = decided.any(axis=1)
    first_idx = np.argmax(decided, axis=1)
    rows = np.arange(n)
    margin_end = diff_running[rows, first_idx]
    remaining_end = remaining[first_idx]

    win_end = ended & (margin_end > 0)
    lose_end = ended & (margin_end < 0)
    tied = ~ended
    n_tied = int(tied.sum())

    n_win = int(win_end.sum()) + int(extra_win[tied].sum())
    n_lose = int(lose_end.sum()) + int((~extra_win[tied] & (extra_hole[tied] != -1)).sum())
    n_halved_extra = int((extra_hole[tied] == -1).sum())  # unresolved after max extras

    result = {
        "n": n,
        "p_win": n_win / n,
        "p_lose": n_lose / n,
        "p_extra": n_tied / n,
        "p_halve18": n_tied / n,
        "expected_points": (n_win + 0.5 * n_halved_extra) / n,
    }

    if not want_scorelines:
        return result

    labels_win = {}
    labels_lose = {}
    win_rows = np.nonzero(win_end)[0]
    for i in win_rows:
        lab = "%d&%d" % (int(margin_end[i]), int(remaining_end[i]))
        if remaining_end[i] <= 0:
            lab = "1UP"
        labels_win[lab] = labels_win.get(lab, 0) + 1
    lose_rows = np.nonzero(lose_end)[0]
    for i in lose_rows:
        lab = "%d&%d" % (int(-margin_end[i]), int(remaining_end[i]))
        if remaining_end[i] <= 0:
            lab = "1UP"
        labels_lose[lab] = labels_lose.get(lab, 0) + 1
    tied_rows = np.nonzero(tied)[0]
    for i in tied_rows:
        h = int(extra_hole[i])
        hole_lab = "21st+" if h >= 21 else "%dth" % h
        if extra_win[i]:
            labels_win[hole_lab] = labels_win.get(hole_lab, 0) + 1
        elif h != -1:
            labels_lose[hole_lab] = labels_lose.get(hole_lab, 0) + 1
    result["scorelines"] = {
        "win": {k: v / n for k, v in sorted(labels_win.items(), key=lambda kv: -kv[1])},
        "lose": {k: v / n for k, v in sorted(labels_lose.items(), key=lambda kv: -kv[1])},
    }
    return result


def _run_match(sa, sb, rng, want_scorelines=False):
    """sa, sb: (n, 18) drawn per-hole relative scores for A and B."""
    diff = np.sign(sb - sa)  # +1: A wins the hole
    running = np.cumsum(diff, axis=1)
    decided, remaining = _decide(running, 18)
    n = running.shape[0]
    tied_idx = np.nonzero(~decided.any(axis=1))[0]
    extra_win = np.zeros(n, dtype=bool)
    extra_hole = np.full(n, -1, dtype=int)
    if tied_idx.size:
        w, h = _resolve_extra(sa, sb, rng, tied_idx.size)
        extra_win[tied_idx] = w
        extra_hole[tied_idx] = h
    return _summarise(running, decided, remaining, extra_win, extra_hole, n, want_scorelines)


def simulate_singles(dist_a, dist_b, n, rng, want_scorelines=False):
    sa = draw_scores(dist_a, rng, n, 18)
    sb = draw_scores(dist_b, rng, n, 18)
    return _run_match(sa, sb, rng, want_scorelines)


def simulate_fourball(dist_a1, dist_a2, dist_b1, dist_b2, n, rng):
    pa = np.minimum(draw_scores(dist_a1, rng, n, 18), draw_scores(dist_a2, rng, n, 18))
    pb = np.minimum(draw_scores(dist_b1, rng, n, 18), draw_scores(dist_b2, rng, n, 18))
    return _run_match(pa, pb, rng)


def simulate_foursomes(dist_pair_a, dist_pair_b, n, rng):
    return simulate_singles(dist_pair_a, dist_pair_b, n, rng)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    d_a = hole_dist(-2.0)
    d_b = hole_dist(0.0)
    res = simulate_singles(d_a, d_b, 50000, rng, want_scorelines=True)
    print("Scheffler-ish vs average tour player, 50k sims:")
    print("  P(win)      %.4f" % res["p_win"])
    print("  P(extra)    %.4f" % res["p_extra"])
    print("  top scorelines:", dict(list(res["scorelines"]["win"].items())[:8]))
    assert abs(res["p_win"] + res["p_lose"] + 0.0 - 1.0) < 0.01
    even = simulate_singles(d_b, d_b, 20000, rng)
    assert abs(even["p_win"] - 0.5) < 0.02, even
    print("  even-match check: P(win)=%.4f OK" % even["p_win"])
