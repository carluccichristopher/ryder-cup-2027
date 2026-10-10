"""Run the full model and write docs/data/model_output.json.

Usage:  python3 -m model.pipeline
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np

from . import config, fetch_espn, matchplay, optimizer, pool as pool_mod, ratings, team_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "data", "model_output.json")

POOL_TOP_N = 18        # candidates per side shipped to the site
TEAM_SIZE = config.TEAM_SIZE


def _clean_player(p):
    headshot = p.get("headshot")
    if not headshot and p.get("espn_id"):
        headshot = "https://a.espncdn.com/i/headshots/golf/players/full/%s.png" % p["espn_id"]
    out = {
        "name": p["name"],
        "team": p["team"],
        "espn_id": p.get("espn_id"),
        "headshot": headshot,
        "country": p.get("country"),
        "owgr_rank": p.get("owgr_rank"),
        "rating": round(p.get("rating", 0.0), 3),
        "prior": round(p.get("prior", 0.0), 3),
        "momentum": round(p.get("momentum", 0.0), 3),
        "n_events": p.get("n_events", 0),
        "apps": p.get("apps", 0),
        "liv": bool(p.get("liv")),
        "note": p.get("note"),
        "points": round(p.get("points", 0.0), 1),
        "points_source": p.get("points_source", "none"),
        "p_team": round(p.get("p_team", 0.0), 4),
        "p_auto": round(p.get("p_auto", 0.0), 4),
        "p_pick": round(p.get("p_pick", 0.0), 4),
        "projected_auto": p.get("projected_auto", False),
        "projected_pick": p.get("projected_pick", False),
        "selection": p.get("selection", {}),
        "seed": p.get("seed", False),
    }
    if p.get("hole_dist"):
        out["hole_dist"] = {k: round(v, 5) for k, v in p["hole_dist"].items()}
    return out


def run(verbose=True):
    t0 = time.time()
    log = lambda *a: print(*a, flush=True) if verbose else None
    today = dt.date.today()
    warnings = []

    # ---------------- data -------------------------------------------------
    log("Fetching OWGR...")
    owgr, owgr_as_of, owgr_src = fetch_espn.fetch_owgr(limit=400)
    log("  %d ranked players (as of %s, %s)" % (len(owgr), owgr_as_of, owgr_src))

    log("Fetching event results (%d weeks)..." % config.FORM_WEEKS_WINDOW)
    events, sources = fetch_espn.fetch_results_window()
    usable = [e for e in events if e["complete"] and e["class"] != "excluded"]
    log("  %d completed events in window" % len(usable))

    manual_events = ratings.load_manual_events(today)
    manual_in_window = [e for e in manual_events
                        if dt.date.fromisoformat(e["date"]) >= today - dt.timedelta(days=config.FORM_WEEKS_WINDOW * 7)]
    log("  %d manual log events in window" % len(manual_in_window))

    # ---------------- pool + ratings ---------------------------------------
    pool, discovered = pool_mod.build_pool(owgr, today)
    log("Pool: %d candidates (%d auto-discovered)" % (len(pool), len(discovered)))

    all_events = usable + manual_in_window
    obs = ratings.collect_observations(all_events, [p["name"] for p in pool])

    official_eu_points, eu_points_as_of = ratings.load_official_europe_points()

    for side in ("USA", "EUR"):
        side_players = pool_mod.side_pool(pool, side)
        names = [p["name"] for p in side_players]
        modeled_us, _ = ratings.estimate_points(names, usable, "USA", today)
        modeled_eu, _ = ratings.estimate_points(names, usable + manual_events, "EUR", today)
        # Europe: official snapshot wins where present.
        for p in side_players:
            if side == "USA":
                p["points"] = modeled_us.get(p["name"], 0.0)
                p["points_source"] = "modeled" if p["points"] > 0 else "none"
            else:
                if p["name"] in official_eu_points:
                    p["points"] = official_eu_points[p["name"]]
                    p["points_source"] = "official"
                else:
                    p["points"] = modeled_eu.get(p["name"], 0.0)
                    p["points_source"] = "modeled" if p["points"] > 0 else "none"
            rating, momentum, n_events = ratings.compute_rating(p, obs.get(p["name"], []), today)
            p["rating"] = rating
            p["momentum"] = momentum
            p["n_events"] = n_events

    # ---------------- selection model ---------------------------------------
    rng = np.random.default_rng(config.SIM_SEED)
    side_info = {}
    for side in ("USA", "EUR"):
        side_players = pool_mod.side_pool(pool, side)
        if len(side_players) < TEAM_SIZE:
            warnings.append("Only %d %s candidates found; teams may be thin." % (len(side_players), side))
        official_points = {p["name"]: p.get("points", 0.0) for p in side_players}
        frac = team_model.selection_scores(side_players, side, today, official_points)
        info = team_model.selection_probabilities(side_players, official_points, side, today, rng)
        info["frac"] = frac
        side_info[side] = info
        side_players.sort(key=lambda p: -p.get("p_team", 0))
        log("  %s race %.0f%% complete; projected auto: %s" %
            (side, 100 * frac, ", ".join(info["projected_auto"])))

    def team_of(side):
        """The projected 12: 6 projected automatic qualifiers + 6 projected picks.
        This mirrors the real selection structure, so points-race leaders make
        the squad even when their raw form rating is modest."""
        sp = pool_mod.side_pool(pool, side)
        by_name = {p["name"]: p for p in sp}
        names = side_info[side]["projected_auto"] + side_info[side]["projected_picks"]
        team = [by_name[n] for n in names if n in by_name]
        if len(team) < TEAM_SIZE:
            have = {p["name"] for p in team}
            rest = sorted([p for p in sp if p["name"] not in have],
                          key=lambda p: -p.get("p_team", 0))
            team.extend(rest[:TEAM_SIZE - len(team)])
        return team

    us_team, eu_team = team_of("USA"), team_of("EUR")

    # hole distributions for everyone we ship (sharpened for match sims)
    for p in pool_mod.side_pool(pool, "USA") + pool_mod.side_pool(pool, "EUR"):
        p["hole_dist"] = matchplay.sim_dist(p.get("rating", 0.0))

    # ---------------- matchups ----------------------------------------------
    log("Simulating 12x12 singles matrix...")
    matrix, details = optimizer.build_matrix(us_team, eu_team, rng)
    lineups = optimizer.optimal_lineups(matrix, us_team, eu_team)

    matrix_out = {
        "us": [p["name"] for p in us_team],
        "eu": [p["name"] for p in eu_team],
        "cells": [[{
            "w": round(matrix[i][j], 4),
            "x": round(details[(i, j)]["p_extra"], 4),
            "l": round(1 - matrix[i][j] - 0, 4),  # matches always decided
        } for j in range(len(eu_team))] for i in range(len(us_team))],
    }

    log("Suggesting pairings...")
    us_avg = float(np.mean([p["rating"] for p in eu_team]))
    eu_avg = float(np.mean([p["rating"] for p in us_team]))
    pairings = {}
    for side, team, opp_avg in (("USA", us_team, us_avg), ("EUR", eu_team, eu_avg)):
        pairings[side] = {
            "fourball": optimizer.suggest_pairs(team, opp_avg, "fourball", rng),
            "foursomes": optimizer.suggest_pairs(team, opp_avg, "foursomes", rng),
        }

    # ---------------- whole-cup simulation -----------------------------------
    log("Simulating full 28-point cups...")
    _, asg_us = optimizer.assign_max(matrix)
    _, asg_eu = optimizer.assign_min(matrix)
    session = optimizer.simulate_cup(
        matrix, pairings["USA"]["fourball"], pairings["EUR"]["fourball"],
        pairings["USA"]["foursomes"], pairings["EUR"]["foursomes"],
        us_team, eu_team, asg_us, asg_eu, rng)

    # ---------------- output -------------------------------------------------
    players_out = {}
    for side in ("USA", "EUR"):
        sp = pool_mod.side_pool(pool, side)
        sp.sort(key=lambda p: -p.get("p_team", 0))
        players_out[side] = [_clean_player(p) for p in sp[:POOL_TOP_N]]

    output = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "today": today.isoformat(),
            "owgr_as_of": owgr_as_of,
            "eu_points_as_of": eu_points_as_of,
            "event": {
                "name": "The 46th Ryder Cup",
                "venue": config.VENUE,
                "start": config.RYDER_CUP_START.isoformat(),
                "end": config.RYDER_CUP_END.isoformat(),
                "us_captain": config.US_CAPTAIN,
                "eu_captain": config.EU_CAPTAIN,
                "us_vice_captains": config.US_VICE_CAPTAINS,
                "eu_vice_captains": config.EU_VICE_CAPTAINS,
            },
            "qualification": {
                "usa": {
                    "window": [config.US_QUAL_START.isoformat(), config.QUAL_END.isoformat()],
                    "fraction_complete": round(side_info["USA"]["frac"], 4),
                    "rules": ("Points from 2027 PGA Tour events only (American Express in January "
                              "through the BMW Championship). Majors 3,000 winner points; THE PLAYERS "
                              "2,850; Signature and Playoff events 2,250; full-field events 1,500. "
                              "Top 6 qualify automatically; 6 captain's picks."),
                },
                "eur": {
                    "window": [config.EU_QUAL_START.isoformat(), config.QUAL_END.isoformat()],
                    "fraction_complete": round(side_info["EUR"]["frac"], 4),
                    "rules": ("Single European points list from 27 Aug 2026 to 22 Aug 2027. "
                              "Majors 5,000-point pools; PGA TOUR Signature/Players/Playoffs 3,000; "
                              "Rolex Series and regular PGA TOUR 2,000; DP World Tour Back 9 1,500; "
                              "Global Series 1,000. Leading six qualify; 6 captain's picks."),
                },
            },
            "sources": {
                "rankings": "ESPN/OWGR (site.web.api.espn.com golf rankings)",
                "results": "ESPN PGA Tour weekly scoreboards",
                "official_europe_points": "Ryder Cup Europe published snapshot (manual file)",
                "disclaimer": ("Fan model for entertainment. Estimated points approximate the "
                               "official schedules; probabilities are model opinion, not betting advice."),
            },
            "data": {
                "events_counted": len(usable),
                "manual_events_counted": len(manual_in_window),
                "fetch_sources": sources,
            },
            "model": {
                "match_sharpness": config.MATCH_SHARPNESS,
                "home_advantage_strokes": config.HOME_ADVANTAGE,
                "form_half_life_weeks": config.FORM_HALF_LIFE_WEEKS,
                "selection_sim_sims": config.SELECTION_SIMS,
            },
            "warnings": warnings,
            "discovered": discovered,
            "runtime_seconds": round(time.time() - t0, 1),
            "python": platform.python_version(),
        },
        "players": players_out,
        "teams": {
            "USA": {
                "captain": config.US_CAPTAIN,
                "projected_auto": side_info["USA"]["projected_auto"],
                "projected_picks": side_info["USA"]["projected_picks"],
            },
            "EUR": {
                "captain": config.EU_CAPTAIN,
                "projected_auto": side_info["EUR"]["projected_auto"],
                "projected_picks": side_info["EUR"]["projected_picks"],
            },
        },
        "matrix": matrix_out,
        "lineups": lineups,
        "pairings": pairings,
        "session": session,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(output, fh, indent=1)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    log("Wrote %s (%.0f KB) in %.1fs" % (OUT_PATH, size_kb, time.time() - t0))
    return output


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print("PIPELINE FAILED:", exc, file=sys.stderr)
        raise
