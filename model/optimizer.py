"""Lineup optimization: assignment problem (Hungarian algorithm) + pairings."""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import config, matchplay


def hungarian(cost):
    """Minimum-cost perfect assignment on a square matrix. O(n^3).

    Returns (total_cost, assignment) where assignment[i] is the chosen column
    for row i. Classic potential-based implementation (Joerin-Volgenant style).
    """
    n = len(cost)
    m = len(cost[0])
    assert n <= m, "hungarian expects cols >= rows"
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)          # p[j]: row matched to column j (1-indexed)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if j1 == -1:
                raise ValueError("hungarian: no feasible assignment")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))
    return total, assignment


def assign_max(matrix):
    """Maximise total. Returns (value, assignment)."""
    cost = [[-matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
    total, assignment = hungarian(cost)
    return -total, assignment


def assign_min(matrix):
    cost = [list(row) for row in matrix]
    return hungarian(cost)


def build_matrix(us_team, eu_team, rng, n_sims=None):
    """12x12 expected-points matrix for US player i vs European player j."""
    n_sims = n_sims or config.MATCH_SIMS_MATRIX
    m = [[0.0] * len(eu_team) for _ in range(len(us_team))]
    details = {}
    for i, a in enumerate(us_team):
        for j, b in enumerate(eu_team):
            res = matchplay.simulate_singles(
                a["hole_dist"], b["hole_dist"], n_sims, rng)
            m[i][j] = res["expected_points"]
            details[(i, j)] = res
    return m, details


def optimal_lineups(matrix, us_team, eu_team):
    """Both captains' ideal singles boards plus per-player best/worst matchups."""
    val_us, asg_us = assign_max(matrix)
    val_eu, asg_eu = assign_min(matrix)
    us_optimal = [{
        "us": us_team[i]["name"],
        "eu": eu_team[j]["name"],
        "p_us_win": matrix[i][j],
    } for i, j in enumerate(asg_us)]
    eu_optimal = [{
        "us": us_team[i]["name"],
        "eu": eu_team[j]["name"],
        "p_us_win": matrix[i][j],
    } for i, j in enumerate(asg_eu)]
    player_best_worst = []
    for i, a in enumerate(us_team):
        row = matrix[i]
        best_j = int(np.argmax(row))
        worst_j = int(np.argmin(row))
        player_best_worst.append({
            "us": a["name"],
            "best": {"vs": eu_team[best_j]["name"], "p_us_win": row[best_j]},
            "worst": {"vs": eu_team[worst_j]["name"], "p_us_win": row[worst_j]},
        })
    for j, b in enumerate(eu_team):
        col = [matrix[i][j] for i in range(len(us_team))]
        best_i = int(np.argmin(col))   # best for Europe = lowest US win prob
        worst_i = int(np.argmax(col))
        player_best_worst.append({
            "eu": b["name"],
            "best": {"vs": us_team[best_i]["name"], "p_eu_win": 1.0 - col[best_i]},
            "worst": {"vs": us_team[worst_i]["name"], "p_eu_win": 1.0 - col[worst_i]},
        })
    return {
        "us_board": us_optimal,
        "eu_board": eu_optimal,
        "us_board_expected_us_pts": val_us,
        "eu_board_expected_us_pts": val_eu,
        "best_worst": player_best_worst,
    }


def suggest_pairs(team, opp_avg_rating, fmt, rng, top_k=4, n_sims=6000):
    """Rank within-team pairs for a format against a synthetic opposing pair."""
    players = team
    pairs = []
    for a, b in itertools.combinations(range(len(players)), 2):
        ra, rb = players[a]["rating"], players[b]["rating"]
        opp = opp_avg_rating
        if fmt == "fourball":
            res = matchplay.simulate_fourball(
                players[a]["hole_dist"], players[b]["hole_dist"],
                matchplay.sim_dist(opp), matchplay.sim_dist(opp),
                n_sims, rng)
        else:  # foursomes
            res = matchplay.simulate_foursomes(
                matchplay.sim_foursomes_dist(ra, rb),
                matchplay.sim_foursomes_dist(opp, opp),
                n_sims, rng)
        pairs.append({
            "i": a, "j": b,
            "players": [players[a]["name"], players[b]["name"]],
            "expected_points": res["expected_points"],
        })
    pairs.sort(key=lambda p: -p["expected_points"])
    chosen, used = [], set()
    for p in pairs:
        if p["i"] in used or p["j"] in used:
            continue
        chosen.append({"players": p["players"], "expected_points": p["expected_points"]})
        used.add(p["i"])
        used.add(p["j"])
        if len(chosen) == top_k:
            break
    return chosen


def _home_shift(p, k):
    """Nudge a win probability toward 0.5 in logit space (home advantage)."""
    tiny = 1e-9
    p = min(max(p, tiny), 1 - tiny)
    z = math.log(p / (1 - p)) - k
    return 1.0 / (1.0 + math.exp(-z))


def simulate_cup(matrix, us_pairs_fourball, eu_pairs_fourball,
                 us_pairs_foursomes, eu_pairs_foursomes,
                 us_team, eu_team, asg_us, asg_eu, rng,
                 n_sims=None):
    """Simulate full 28-point Ryder Cups.

    Friday and Saturday each play 4 four-ball and 4 foursomes matches using the
    suggested pairings (each pairing plays twice), then 12 singles whose board
    alternates between each captain's ideal assignment. Europe gets a small home
    advantage at Adare Manor (config.HOME_ADVANTAGE).
    """
    n_sims = n_sims or config.SESSION_SIMS
    adv = config.HOME_ADVANTAGE

    def eu_rating(team, name):
        return _rating(team, name) - adv  # home side plays better

    team_probs = []
    for _day in ("friday", "saturday"):
        for k in range(len(us_pairs_fourball)):
            uf = us_pairs_fourball[k]["players"]
            ef = eu_pairs_fourball[k]["players"]
            us_d = [matchplay.sim_dist(_rating(us_team, n)) for n in uf]
            eu_d = [matchplay.sim_dist(eu_rating(eu_team, n)) for n in ef]
            res = matchplay.simulate_fourball(us_d[0], us_d[1], eu_d[0], eu_d[1], 4000, rng)
            team_probs.append(res["expected_points"])
        for k in range(len(us_pairs_foursomes)):
            uf = us_pairs_foursomes[k]["players"]
            ef = eu_pairs_foursomes[k]["players"]
            us_r = sum(_rating(us_team, n) for n in uf) / 2.0
            eu_r = sum(eu_rating(eu_team, n) for n in ef) / 2.0
            res = matchplay.simulate_foursomes(
                matchplay.sim_foursomes_dist(us_r, us_r),
                matchplay.sim_foursomes_dist(eu_r, eu_r), 4000, rng)
            team_probs.append(res["expected_points"])

    home_k = 0.15  # logit nudge for the home side in singles
    singles_us_board = [_home_shift(matrix[i][j], home_k) for i, j in enumerate(asg_us)]
    singles_eu_board = [_home_shift(matrix[i][j], home_k) for i, j in enumerate(asg_eu)]

    draws = rng.random((n_sims, len(team_probs) + 12))
    team_pts = (draws[:, :len(team_probs)] < np.array(team_probs)).sum(axis=1)
    half_us = n_sims // 2
    singles_pts = np.empty(n_sims)
    for block, probs in ((slice(0, half_us), singles_us_board), (slice(half_us, n_sims), singles_eu_board)):
        singles_pts[block] = (draws[block, len(team_probs):] < np.array(probs)).sum(axis=1)
    us_total = team_pts + singles_pts

    dist = {}
    for t in us_total:
        dist[int(t)] = dist.get(int(t), 0) + 1
    dist = {k: v / n_sims for k, v in dist.items()}
    p_usa = float(np.mean(us_total > 14))
    p_eur = float(np.mean(us_total < 14))
    p_tie = float(np.mean(us_total == 14))
    return {
        "n": n_sims,
        "p_usa_win": p_usa,
        "p_europe_win": p_eur,
        "p_14_14": p_tie,  # tie -> Europe retains
        "p_europe_retain": p_eur + p_tie,
        "median_us_points": float(np.median(us_total)),
        "mean_us_points": float(np.mean(us_total)),
        "us_points_dist": {str(k): v for k, v in sorted(dist.items()) if v > 0.0005},
    }


def _rating(team, name):
    for p in team:
        if p["name"] == name:
            return p["rating"]
    return 0.0


if __name__ == "__main__":
    # Self-test: brute force check on small random matrices
    import itertools as it
    import random as _random

    _random.seed(1)
    for trial in range(200):
        n = _random.randint(2, 6)
        cost = [[_random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
        best = min(sum(cost[i][p[i]] for i in range(n))
                   for p in it.permutations(range(n)))
        total, asg = hungarian(cost)
        got = sum(cost[i][asg[i]] for i in range(n))
        assert abs(got - best) < 1e-9, (trial, got, best)
    print("hungarian self-test: 200 trials OK")
