"""Build the candidate pool: hand-curated seed players merged with live OWGR.

The seed pool (data/seed_pool.json) is the authoritative team classification:
it forces in notable candidates (including LIV players who don't always appear
in live feeds) and encodes Ryder Cup appearances. Live OWGR data then refreshes
ranks, ids, countries and headshots, and auto-discovers any top-100
American/European not already in the seed file.
"""
from __future__ import annotations

import json
import os

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_POOL_PATH = os.path.join(ROOT, "data", "seed_pool.json")
OVERRIDES_PATH = os.path.join(ROOT, "data", "manual", "overrides.json")


def _load(path, default):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def rank_prior(rank) -> float:
    """Interpolate the OWGR rank onto a strokes-vs-field prior."""
    if rank is None:
        return 0.30
    anchors = config.RANK_PRIOR_ANCHORS
    if rank <= anchors[0][0]:
        return anchors[0][1]
    for (r1, a1), (r2, a2) in zip(anchors, anchors[1:]):
        if r1 <= rank <= r2:
            if r2 == r1:
                return a1
            return a1 + (a2 - a1) * (rank - r1) / float(r2 - r1)
    return anchors[-1][1]


def classify_country(country):
    if country in ("USA", "United States", "U.S."):
        return "USA"
    if country in config.EUROPE_COUNTRIES:
        return "EUR"
    return None


def _norm_name(name):
    return (name or "").strip().lower().replace(".", "").replace("  ", " ")


def build_pool(owgr_players, today):
    """Return (pool, discovered) where pool is a list of candidate dicts."""
    seed = _load(SEED_POOL_PATH, {"players": [], "note": ""})
    overrides = _load(OVERRIDES_PATH, {"players": []})
    override_by_name = {_norm_name(p.get("name")): p for p in overrides.get("players", [])}

    pool_by_key = {}

    # 1. Seed entries: authoritative team assignment.
    for s in seed.get("players", []):
        p = {
            "name": s["name"],
            "team": s.get("team"),
            "espn_id": s.get("espn_id"),
            "headshot": None,
            "country": s.get("country"),
            "owgr_rank": None,
            "prior_rank": None,
            "apps": int(s.get("apps") or 0),
            "liv": bool(s.get("liv")),
            "note": s.get("note"),
            "manual_rating": s.get("manual_rating"),
            "ineligible": bool(s.get("ineligible")),
            "seed": True,
        }
        pool_by_key[_norm_name(p["name"])] = p

    # 2. Merge live OWGR.
    for w in owgr_players:
        key = _norm_name(w["name"])
        if key in pool_by_key:
            p = pool_by_key[key]
            p["owgr_rank"] = w.get("rank")
            p["country"] = w.get("country") or p.get("country")
            if w.get("espn_id"):
                p["espn_id"] = w["espn_id"]
            p["headshot"] = w.get("headshot")
        else:
            team = classify_country(w.get("country"))
            if team and w.get("rank") and w["rank"] <= config.TOP_DISCOVERY_RANK:
                pool_by_key[key] = {
                    "name": w["name"],
                    "team": team,
                    "espn_id": w.get("espn_id"),
                    "headshot": w.get("headshot"),
                    "country": w.get("country"),
                    "owgr_rank": w.get("rank"),
                    "prior_rank": w.get("rank"),
                    "apps": 0,
                    "liv": False,
                    "note": None,
                    "manual_rating": None,
                    "ineligible": False,
                    "seed": False,
                }

    # 3. Manual overrides (eligibility, notes, ratings).
    for key, ov in override_by_name.items():
        if key in pool_by_key:
            p = pool_by_key[key]
            for field in ("team", "apps", "liv", "note", "manual_rating", "ineligible"):
                if field in ov:
                    p[field] = ov[field]

    # 4. Priors from OWGR rank (or manual override).
    discovered = []
    for p in pool_by_key.values():
        if p.get("manual_rating") is not None:
            p["prior"] = float(p["manual_rating"])
        else:
            p["prior"] = rank_prior(p.get("owgr_rank"))
        p["prior_rank"] = p.get("owgr_rank")
        if not p.get("team"):
            p["ineligible"] = True  # seed player with unknown team: ignore in selection
        if not p["seed"]:
            discovered.append(p["name"])

    pool = [p for p in pool_by_key.values() if p.get("team")]
    pool.sort(key=lambda x: (x["team"], x["owgr_rank"] or 9999, x["name"]))
    return pool, sorted(discovered)


def side_pool(pool, side):
    return [p for p in pool if p["team"] == side and not p.get("ineligible")]
