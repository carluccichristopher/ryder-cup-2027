"""Fetch live golf data from ESPN's public endpoints, with a disk cache.

Everything the model knows about the world flows through here:
  - OWGR world ranking (top 1000, with ESPN athlete ids, flags, headshots)
  - PGA Tour weekly scoreboards (which embed full final leaderboards)

If the network is unavailable the fetcher transparently falls back to the last
cached copy, so the pipeline (and the GitHub Action) never hard-fails.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "cache")

RANKINGS_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/golf/all/rankings"
    "?region=us&lang=en&polls=1"
)
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/{league}/scoreboard?dates={d1}-{d2}"

# ESPN league slugs we try for extra tours. As of Oct 2026 only 'pga' responds
# on this endpoint; the others are future-proofing and fail silently.
LEAGUE_SLUGS = ["pga"]
EXTRA_LEAGUE_SLUGS = ["euro", "european-tour", "dpwt"]

USER_AGENT = "ryder-cup-2027-predictor (personal GitHub Pages project)"
TTL_HOURS = 144  # 6 days: the weekly Action refreshes, mid-week local runs reuse


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + ".json")


def _http_get_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "identity",  # avoid gzip handling surprises
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(gzip.decompress(raw).decode("utf-8"))


def get_json(url: str, cache_key: str, ttl_hours: int = TTL_HOURS, timeout: int = 30):
    """GET JSON with cache-first behaviour. Returns (payload, source) where
    source is one of 'net', 'cache', 'stale-cache'."""
    path = _cache_path(cache_key)
    if os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h < ttl_hours:
            with open(path) as fh:
                return json.load(fh), "cache"
    try:
        payload = _http_get_json(url, timeout=timeout)
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return payload, "net"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh), "stale-cache"
        raise


# ----------------------------------------------------------------------------
# Rankings
# ----------------------------------------------------------------------------

def fetch_owgr(limit: int = 300):
    """Return (players, as_of) from the Official World Golf Ranking.

    Each player dict: rank, prev_rank, name, espn_id, country, headshot.
    """
    data, src = get_json(RANKINGS_URL, "owgr")
    players = []
    as_of = None
    for ranking in data.get("rankings") or []:
        if ranking.get("type") != "WORLDRANK":
            continue
        as_of = (ranking.get("update") or "").replace("/", "-")
        for entry in (ranking.get("ranks") or [])[:limit]:
            ath = entry.get("athlete") or {}
            flag = ath.get("flag") or {}
            birth = ath.get("birthPlace") or {}
            players.append({
                "rank": entry.get("current"),
                "prev_rank": entry.get("previous"),
                "name": ath.get("fullName") or ath.get("displayName"),
                "espn_id": str(ath.get("id")) if ath.get("id") else None,
                "country": flag.get("alt") or birth.get("countryAbbreviation"),
                "headshot": (ath.get("headshot") or {}).get("href"),
            })
    if not players:
        raise ValueError("OWGR payload contained no players (source=%s)" % src)
    return players, as_of, src


# ----------------------------------------------------------------------------
# Weekly event results
# ----------------------------------------------------------------------------

def _to_par(display_value) -> float:
    """'+6' -> 6.0, '-10' -> -10.0, 'E' -> 0.0, 'None' -> None."""
    if display_value is None:
        return None
    s = str(display_value).strip()
    if s in ("E", "", "-", "--"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _parse_competitor(comp: dict) -> dict:
    ath = comp.get("athlete") or {}
    if not isinstance(ath, dict):
        return {"name": None}
    flag = ath.get("flag") or {}
    if not isinstance(flag, dict):
        flag = {}
    score = comp.get("score")
    total_strokes = to_par = None
    if isinstance(score, dict):
        total_strokes = score.get("value")
        to_par = _to_par(score.get("displayValue"))
    elif isinstance(score, str):
        to_par = _to_par(score)
    status = comp.get("status") or {}
    if not isinstance(status, dict):
        status = {}
    stype = ((status.get("type") or {}).get("name") or "").upper()
    rounds = []
    for ls in comp.get("linescores") or []:
        if not isinstance(ls, dict):
            continue
        val = ls.get("value")
        if val not in (None, 0.0):
            try:
                rounds.append(float(val))
            except (TypeError, ValueError):
                pass
    pos_str = ((status.get("position") or {}).get("displayName") or "").strip()
    position = None
    if pos_str and pos_str not in ("-", ""):
        try:
            position = int(pos_str.lstrip("T"))
        except ValueError:
            position = None
    return {
        "espn_id": str(ath.get("id")) if ath.get("id") else None,
        "name": ath.get("fullName") or ath.get("displayName"),
        "country": flag.get("alt"),
        "headshot": (ath.get("headshot") or {}).get("href") if isinstance(ath.get("headshot"), dict) else None,
        "total_strokes": total_strokes,
        "to_par": to_par,
        "status": stype or status.get("displayValue"),
        "position": position,
        "rounds": rounds,
    }


def fetch_week(league: str, week_start: dt.date, days: int = 7, ttl_hours: int = TTL_HOURS):
    """Fetch one week of a tour's scoreboard. Returns (events, source).

    Each event: {id, name, date, class, status, competitors}. Completed events
    include the full final leaderboard (positions relative to the field).
    """
    d1 = week_start.strftime("%Y%m%d")
    d2 = (week_start + dt.timedelta(days=days - 1)).strftime("%Y%m%d")
    url = SCOREBOARD_URL.format(league=league, d1=d1, d2=d2)
    key = "week-%s-%s" % (league, week_start.isoformat())
    try:
        data, src = get_json(url, key, ttl_hours=ttl_hours)
    except Exception:
        return [], "unavailable"
    events = []
    for ev in data.get("events") or []:
        state = ((ev.get("status") or {}).get("type") or {}).get("state")
        name = ev.get("name") or ""
        cls = config.classify_event(name)
        if cls == "excluded":
            continue
        comps = []
        for comp in (ev.get("competitions") or [{}])[0].get("competitors") or []:
            try:
                c = _parse_competitor(comp)
            except Exception:
                continue
            if c["name"] and (c["total_strokes"] is not None or c["to_par"] is not None):
                comps.append(c)
        events.append({
            "id": ev.get("id"),
            "name": name,
            "date": (ev.get("endDate") or ev.get("date") or "")[:10],
            "class": cls,
            "state": state,
            "complete": state == "post",
            "competitors": comps,
        })
    return events, src


def fetch_results_window(weeks: int = None, end_date: dt.date = None):
    """Fetch all tour events in the trailing window. Returns (events, sources)."""
    weeks = weeks or config.FORM_WEEKS_WINDOW
    end = end_date or dt.date.today()
    # Anchor weeks on Sundays.
    days_since_sunday = (end.weekday() + 1) % 7
    last_sunday = end - dt.timedelta(days=days_since_sunday)

    # Probe extra league slugs once; skip silently if ESPN doesn't serve them.
    active = []
    for league in LEAGUE_SLUGS + EXTRA_LEAGUE_SLUGS:
        if league in LEAGUE_SLUGS:
            active.append(league)
            continue
        probe, src = fetch_week(league, last_sunday, ttl_hours=1)
        if src != "unavailable":
            active.append(league)

    all_events = []
    sources = {}
    for league in active:
        for i in range(weeks):
            start = last_sunday - dt.timedelta(days=7 * i)
            events, src = fetch_week(league, start)
            sources["%s-%s" % (league, src)] = sources.get("%s-%s" % (league, src), 0) + 1
            all_events.extend(events)
    return all_events, sources
