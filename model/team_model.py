"""Team selection model: who makes each 12-man squad, and how surely?

Each candidate gets a selection score blending five signals:
  form      recency-weighted strokes vs field (from live results)
  rank      OWGR-derived prior
  points    official/modeled Ryder Cup qualifying points
  experience Ryder Cup appearances (captains like known quantities)
  momentum  last-10-weeks form surge

As the qualification race matures, weight flows from form/rank to points, so
real 2027 results gradually take over the model. A Monte Carlo of the selection
process (6 automatic spots by projected final points, then 6 captain's picks
favoring experience) turns scores into probabilities.
"""
from __future__ import annotations

import numpy as np

from . import config

# Rough magnitude of the points still to be awarded in a full race, per side.
RACE_SCALE = {"USA": 4300.0, "EUR": 2400.0}
SOFTMAX_TEMP = 0.35


def window_fraction(side, today):
    start = config.US_QUAL_START if side == "USA" else config.EU_QUAL_START
    if today <= start:
        return 0.0
    if today >= config.QUAL_END:
        return 1.0
    return (today - start).days / float((config.QUAL_END - start).days)


def _weights(frac):
    return {k: config.W_EARLY[k] + (config.W_LATE[k] - config.W_EARLY[k]) * frac
            for k in config.W_EARLY}


def selection_scores(players, side, today, official_points):
    """Attach a selection score + component breakdown to each player dict."""
    frac = window_fraction(side, today)
    w = _weights(frac)

    max_pts = max([official_points.get(p["name"], 0.0) for p in players] or [0.0])
    for p in players:
        pts = official_points.get(p["name"], 0.0)
        pts_share = (pts / max_pts) if max_pts > 0 else 0.0
        comps = {
            "form": -p.get("rating", 0.0),        # negative rating = good
            "rank": -p.get("prior", 0.0),
            "points": min(pts_share, 1.25),
            "experience": min(1.0, (p.get("apps") or 0) / 5.0),
            "momentum": float(np.clip(-p.get("momentum", 0.0), -1.5, 1.5)),
        }
        score = sum(w[k] * comps[k] for k in w)
        p["selection"] = {
            "score": round(score, 4),
            "components": {k: round(v, 3) for k, v in comps.items()},
            "points_share": round(pts_share, 3),
            "weights": {k: round(v, 3) for k, v in w.items()},
        }
    return frac


def project_final_points(players, official_points, side, frac):
    """Projected end-of-race points: current + remaining share via softmax."""
    scale = RACE_SCALE[side]
    scores = np.array([p["selection"]["score"] for p in players])
    cur = np.array([official_points.get(p["name"], 0.0) for p in players])
    e = np.exp((scores - scores.max()) / SOFTMAX_TEMP)
    share = e / e.sum()
    proj = cur + (1.0 - frac) * scale * share
    return np.maximum(proj, cur)


def selection_probabilities(players, official_points, side, today, rng, n_sims=None):
    """Monte Carlo the 12-man selection. Attaches p_auto/p_pick/p_team and a
    deterministic most-likely squad."""
    n_sims = n_sims or config.SELECTION_SIMS
    frac = window_fraction(side, today)
    n = len(players)
    if n == 0:
        return {"frac": frac, "projected_auto": [], "projected_picks": []}

    proj = project_final_points(players, official_points, side, frac)
    scores = np.array([p["selection"]["score"] for p in players])
    apps = np.array([min(1.0, (p.get("apps") or 0) / 4.0) for p in players])

    # --- automatic spots: top 6 by noisy projected final points
    sigma = config.POINTS_NOISE_FRACTION * RACE_SCALE[side] * max(0.03, 1.0 - 0.85 * frac)
    noisy_pts = proj[None, :] + rng.normal(0.0, sigma, size=(n_sims, n))
    order = np.argsort(-noisy_pts, axis=1)
    auto_mask = np.zeros((n_sims, n), dtype=bool)
    for k in range(min(config.AUTO_SPOTS, n)):
        auto_mask[np.arange(n_sims), order[:, k]] = True
    p_auto = auto_mask.mean(axis=0)

    # --- captain's picks: best remaining by experience-tilted score
    pick_scores = scores + config.PICK_EXPERIENCE_BONUS * apps
    pick_total = pick_scores[None, :] + rng.normal(0.0, config.SELECTION_NOISE, size=(n_sims, n))
    pick_total[auto_mask] = -1e9
    pick_order = np.argsort(-pick_total, axis=1)
    pick_mask = np.zeros((n_sims, n), dtype=bool)
    for k in range(min(config.CAPTAIN_PICKS, n)):
        pick_mask[np.arange(n_sims), pick_order[:, k]] = True
    p_pick = pick_mask.mean(axis=0)
    p_team = np.clip(p_auto + pick_mask.mean(axis=0) - (auto_mask & pick_mask).mean(axis=0), 0.0, 1.0)

    for i, p in enumerate(players):
        p["p_auto"] = float(p_auto[i])
        p["p_pick"] = float(p_pick[i])
        p["p_team"] = float(p_team[i])
        p["projected_final_points"] = float(proj[i])

    # Deterministic "most likely" squad.
    auto_idx = list(np.argsort(-proj)[:min(config.AUTO_SPOTS, n)])
    remaining = [i for i in range(n) if i not in auto_idx]
    pick_idx = sorted(remaining, key=lambda i: -pick_scores[i])[:min(config.CAPTAIN_PICKS, len(remaining))]
    for p in players:
        p["projected_auto"] = False
        p["projected_pick"] = False
    for i in auto_idx:
        players[i]["projected_auto"] = True
    for i in pick_idx:
        players[i]["projected_pick"] = True

    return {
        "frac": frac,
        "projected_auto": [players[i]["name"] for i in auto_idx],
        "projected_picks": [players[i]["name"] for i in pick_idx],
    }
