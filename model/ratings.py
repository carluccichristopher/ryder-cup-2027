"""Form ratings and qualification points from live event results.

Rating = strokes per round vs the average player in that week's field,
recency-weighted with a half-life, then shrunk toward the OWGR-derived prior.
Points = official manual snapshots where available, otherwise modeled from
the officially announced point schedules applied to observed results.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_LOG_PATH = os.path.join(ROOT, "data", "manual", "results_log.json")
EUROPE_POINTS_PATH = os.path.join(ROOT, "data", "manual", "europe_points.json")


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def _rounds_from_status(status: str) -> int:
    s = (status or "").upper()
    if "CUT" in s:
        return 2
    if "WD" in s or "DQ" in s:
        return 0
    return 4


def collect_observations(events, pool_names):
    """Return {player_name: [observation, ...]}.

    Observation: {date, event, class, delta, rounds, manual}
    delta = mean per-round strokes vs that field's per-round average.
    """
    obs = {name: [] for name in pool_names}
    for ev in events:
        if not ev.get("complete") or ev.get("class") == "excluded":
            continue
        comps = [c for c in ev.get("competitors", []) if c.get("name")]
        if len(comps) < 15:
            continue

        # Path 1: per-round strokes (preferred).
        max_rounds = 0
        for c in comps:
            max_rounds = max(max_rounds, len(c.get("rounds") or []))
        round_avgs = []
        for r in range(max_rounds):
            vals = [c["rounds"][r] for c in comps
                    if len(c.get("rounds") or []) > r and c["rounds"][r]]
            round_avgs.append(sum(vals) / len(vals) if vals else None)

        # Path 2 fallback: total vs par (finishers only, normalised per round).
        finisher_pars = [c["to_par"] for c in comps
                         if c.get("to_par") is not None
                         and _rounds_from_status(c.get("status") or "F") == 4]
        field_round_par = (sum(finisher_pars) / len(finisher_pars) / 4.0) if finisher_pars else None

        for c in comps:
            name = c["name"]
            if name not in obs:
                continue
            if c.get("assumed_delta") is not None:  # manual log entry
                obs[name].append({
                    "date": ev["date"], "event": ev["name"], "class": ev["class"],
                    "delta": float(c["assumed_delta"]), "rounds": 4, "manual": True,
                })
                continue
            rounds = c.get("rounds") or []
            if max_rounds and rounds:
                deltas = []
                for r in range(min(len(rounds), max_rounds)):
                    if round_avgs[r] is not None and rounds[r]:
                        deltas.append(rounds[r] - round_avgs[r])
                if len(deltas) >= 1:
                    obs[name].append({
                        "date": ev["date"], "event": ev["name"], "class": ev["class"],
                        "delta": sum(deltas) / len(deltas), "rounds": len(deltas),
                        "manual": False,
                    })
                    continue
            if field_round_par is not None and c.get("to_par") is not None:
                nr = _rounds_from_status(c.get("status") or "F")
                if nr >= 2:
                    obs[name].append({
                        "date": ev["date"], "event": ev["name"], "class": ev["class"],
                        "delta": c["to_par"] / nr - field_round_par, "rounds": nr,
                        "manual": False,
                    })
    return obs


def load_manual_events(today):
    """Parse data/manual/results_log.json into synthetic event dicts."""
    if not os.path.exists(RESULTS_LOG_PATH):
        return []
    with open(RESULTS_LOG_PATH) as fh:
        log = json.load(fh)
    events = []
    for ev in log.get("events", []):
        try:
            date = dt.date.fromisoformat(str(ev.get("date"))[:10])
        except ValueError:
            continue
        cls = ev.get("class") or config.classify_event(ev.get("name", ""))
        if cls == "excluded":
            continue
        comps = []
        for r in ev.get("results", []):
            pos = int(r.get("position") or 1)
            # Rough assumption: a winner of a professional event plays about
            # 2.1 strokes/round better than the field; each finishing spot
            # below adds ~0.3. Flagged manual + low confidence weight.
            assumed = -2.1 + 0.30 * (pos - 1)
            comps.append({
                "name": r.get("name"),
                "position": pos,
                "assumed_delta": assumed,
                "status": "F",
            })
        events.append({
            "id": "manual-%s" % ev.get("name", "").replace(" ", "-"),
            "name": ev.get("name"),
            "date": date.isoformat(),
            "class": cls,
            "complete": True,
            "manual": True,
            "competitors": comps,
        })
    return events


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def _obs_weight(obs, today):
    try:
        d = dt.date.fromisoformat(str(obs["date"])[:10])
    except ValueError:
        return 0.0
    weeks = max(0.0, (today - d).days / 7.0)
    if weeks > config.FORM_WEEKS_WINDOW + 8:
        return 0.0
    recency = 0.5 ** (weeks / config.FORM_HALF_LIFE_WEEKS)
    cls_w = config.CLASS_FORM_WEIGHT.get(obs["class"], 0.9)
    rounds_w = min(obs.get("rounds", 2), 4) / 4.0
    manual_w = config.MANUAL_EVENT_WEIGHT if obs.get("manual") else 1.0
    return recency * cls_w * rounds_w * manual_w


def compute_rating(player, observations, today):
    """Blend observed deltas with the prior. Returns (rating, momentum, n_events)."""
    prior = float(player.get("prior") or 0.0)
    w_sum = 0.0
    w_delta = 0.0
    recent_w = 0.0
    recent_d = 0.0
    n_events = 0
    for obs in observations:
        w = _obs_weight(obs, today)
        if w <= 0:
            continue
        w_sum += w
        w_delta += w * obs["delta"]
        n_events += 1
        try:
            d = dt.date.fromisoformat(str(obs["date"])[:10])
            if (today - d).days <= config.MOMENTUM_WINDOW_WEEKS * 7:
                recent_w += w
                recent_d += w * obs["delta"]
        except ValueError:
            pass
    if w_sum <= 0:
        rating = prior
    else:
        blend = w_delta / w_sum
        k = config.PRIOR_STRENGTH
        rating = (w_delta + k * prior) / (w_sum + k)
    momentum = (recent_d / recent_w) if recent_w > 0 else 0.0
    # The field a player beats is tour-relative; ratings are on the PGA scale.
    rating = max(-3.2, min(1.5, rating))
    momentum = max(-2.5, min(2.5, momentum))
    return rating, momentum, n_events


# ---------------------------------------------------------------------------
# Qualification points
# ---------------------------------------------------------------------------

def _share_points(base, pool_mode, position):
    if position is None or position < 1 or position > len(config.POSITION_SHARES):
        return 0.0
    share = config.POSITION_SHARES[position - 1]
    if pool_mode == "us":
        return base * share / config.POSITION_SHARES[0]
    return base * share  # Europe: shares already normalised by the caller


def load_official_europe_points():
    """Official European points snapshot (manual file). {name: (points, as_of)}"""
    if not os.path.exists(EUROPE_POINTS_PATH):
        return {}, None
    with open(EUROPE_POINTS_PATH) as fh:
        data = json.load(fh)
    out = {}
    for row in data.get("points", []):
        if row.get("points") is not None:
            out[row["name"]] = float(row["points"])
    return out, data.get("as_of")


def estimate_points(pool_names, events, side, today):
    """Model each player's qualifying points from observed results.

    Returns ({name: points}, {name: n_events_counted}).
    Official manual snapshots are merged by the caller (Europe only).
    """
    start = config.US_QUAL_START if side == "USA" else config.EU_QUAL_START
    pts = {name: 0.0 for name in pool_names}
    counted = {name: 0 for name in pool_names}
    for ev in events:
        if not ev.get("complete") or ev.get("class") == "excluded":
            continue
        try:
            d = dt.date.fromisoformat(str(ev["date"])[:10])
        except ValueError:
            continue
        if d < start or d > today:
            continue
        cls = ev["class"]
        if side == "USA":
            base = config.US_WINNER_POINTS.get(cls)
            pool_mode = "us"
        else:
            base = config.EU_EVENT_POOLS.get(cls)
            pool_mode = "eu"
        if not base:
            continue
        n_shares = min(30, max(1, len(ev.get("competitors", []))))
        # Europe: the pool is shared across the whole field, so normalise the
        # position curve over a standard 30-place payout table (~86% of pool).
        denom = sum(config.POSITION_SHARES[:30]) if pool_mode == "eu" else config.POSITION_SHARES[0]
        for c in ev.get("competitors", []):
            name = c.get("name")
            if name not in pts:
                continue
            pos = c.get("position")
            if pos is None or pos < 1 or pos > 30:
                continue
            if "CUT" in (c.get("status") or "").upper():
                continue
            share = config.POSITION_SHARES[pos - 1]
            if pool_mode == "us":
                gained = base * share / config.POSITION_SHARES[0]
            else:
                gained = base * share / denom
            pts[name] += gained
            counted[name] += 1
    return pts, counted
