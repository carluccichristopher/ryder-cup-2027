"""Central configuration for the 2027 Ryder Cup prediction model.

All competition facts encoded here were verified in October 2026 from official
announcements by the PGA of America and Ryder Cup Europe:
  - 46th Ryder Cup: 17-19 September 2027, Adare Manor, County Limerick, Ireland
  - USA captain Jim Furyk; Europe captain Luke Donald
  - USA: points only from 2027 events (AmEx through BMW Championship), 6 auto + 6 picks
  - Europe: single points list from Aug 27 2026 through Aug 22 2027, 6 auto + 6 picks
"""
from __future__ import annotations

import datetime as dt

# --- The event --------------------------------------------------------------
RYDER_CUP_START = dt.date(2027, 9, 17)
RYDER_CUP_END = dt.date(2027, 9, 19)
VENUE = "Adare Manor, County Limerick, Ireland"
US_CAPTAIN = "Jim Furyk"
EU_CAPTAIN = "Luke Donald"
US_VICE_CAPTAINS = ["Stewart Cink", "Justin Leonard"]
EU_VICE_CAPTAINS = ["Edoardo Molinari", "Francesco Molinari", "Padraig Harrington"]

TEAM_SIZE = 12
AUTO_SPOTS = 6
CAPTAIN_PICKS = 6
POINTS_TO_WIN = 14.5  # 28 matches: first to 14.5 wins; 14-14 means Europe retains

# --- Qualification windows ----------------------------------------------------
US_QUAL_START = dt.date(2027, 1, 18)  # The American Express tournament week
EU_QUAL_START = dt.date(2026, 8, 27)   # Betfred British Masters
QUAL_END = dt.date(2027, 8, 22)        # BMW Championship (concludes both races)

# --- Official point schedules ---------------------------------------------------
# USA (announced 2026-09-09): fixed winner points; scale down a position curve.
US_WINNER_POINTS = {
    "major": 3000,
    "players": 2850,
    "signature": 2250,
    "playoff": 2250,
    "full": 1500,
}
# Europe (announced 2026-08-04): event point POOLS shared by finishing position.
EU_EVENT_POOLS = {
    "major": 5000,
    "players": 3000,
    "signature": 3000,
    "playoff": 3000,
    "rolex": 2000,
    "pga_regular": 2000,
    "back9": 1500,
    "global": 1000,
    "opposite": 1000,
}

# Share of a winner's points (USA) or of the event pool (Europe) awarded to each
# finishing position. Approximation of the official schedules; the pipeline
# labels estimated points as "modeled" versus "official" manual snapshots.
POSITION_SHARES = [
    0.180, 0.110, 0.070, 0.053, 0.043, 0.036, 0.030, 0.026, 0.023, 0.021,
    0.019, 0.017, 0.015, 0.013, 0.011, 0.009, 0.008, 0.007, 0.006, 0.005,
    0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001, 0.001, 0.001,
]

# --- Event classification ---------------------------------------------------------
MAJOR_NAMES = {"Masters Tournament", "PGA Championship", "U.S. Open", "The Open"}
EXCLUDED_EVENTS = {
    "Presidents Cup", "Ryder Cup", "Solheim Cup",
    "Zurich Classic of New Orleans",          # two-man team stroke play
    "PGA TOUR Q-School presented by Korn Ferry",
    "Hero World Challenge",                    # unofficial small field
    "Grant Thornton Invitational",            # mixed-team event
}
EVENT_CLASS = {
    "THE PLAYERS Championship": "players",
    "FedEx St. Jude Championship": "playoff",
    "BMW Championship": "playoff",
    "TOUR Championship": "playoff",
    "The Sentry": "signature",
    "AT&T Pebble Beach Pro-Am": "signature",
    "The Genesis Invitational": "signature",
    "Arnold Palmer Invitational pres. by Mastercard": "signature",
    "RBC Heritage": "signature",
    "Truist Championship": "signature",
    "the Memorial Tournament pres. by Workday": "signature",
    "Travelers Championship": "signature",
    "Cadillac Championship": "signature",
    # DP World Tour (arrive via the manual results log / future feed support)
    "Genesis Scottish Open": "rolex",
    "BMW PGA Championship": "rolex",
    "Horizon Irish Open": "rolex",
    "Irish Open": "rolex",
    "Abu Dhabi HSBC Championship": "rolex",
    "Dubai Desert Classic": "rolex",
    "Betfred British Masters": "back9",
    "FedEx Open de France": "back9",
    "Open de France": "back9",
    "Italian Open": "back9",
    "Spanish Open": "back9",
    "BMW International Open": "back9",
    "KLM Open": "global",
    "Made in Denmark": "global",
    "Alfred Dunhill Links Championship": "global",
}
OPPOSITE_FIELD_TOKENS = (
    "Puerto Rico Open",
    "Myrtle Beach Classic",
    "ISCO Championship",
    "Corales Puntacana",
    "Barracuda",
    "Butterfield",
)
DEFAULT_CLASS = "full"  # unrecognised PGA Tour events

# How much an event's result tells us about a player (field strength proxy).
CLASS_FORM_WEIGHT = {
    "major": 1.35,
    "players": 1.20,
    "signature": 1.15,
    "playoff": 1.15,
    "rolex": 1.05,
    "full": 1.00,
    "pga_regular": 1.00,
    "back9": 0.80,
    "global": 0.65,
    "opposite": 0.55,
}


def classify_event(name: str) -> str:
    """Map an event name to its class, or 'excluded' if not an individual stroke-play event."""
    n = name.strip()
    if n in EXCLUDED_EVENTS:
        return "excluded"
    if n in MAJOR_NAMES:
        return "major"
    if n in EVENT_CLASS:
        return EVENT_CLASS[n]
    for token in OPPOSITE_FIELD_TOKENS:
        if token.lower() in n.lower():
            return "opposite"
    return DEFAULT_CLASS


# --- Ratings ----------------------------------------------------------------------
FORM_WEEKS_WINDOW = 68        # ~16 months of results feed the form rating
FORM_HALF_LIFE_WEEKS = 26.0   # recency decay
PRIOR_STRENGTH = 3.0          # pseudo-observations of prior weight in the blend
MOMENTUM_WINDOW_WEEKS = 10    # "recent form" window

# OWGR rank -> prior ability, in strokes per round vs an average PGA Tour field.
RANK_PRIOR_ANCHORS = [
    (1, -2.15), (3, -1.75), (5, -1.35), (10, -1.00), (15, -0.80),
    (20, -0.62), (30, -0.42), (50, -0.18), (75, 0.02),
    (100, 0.18), (150, 0.40), (250, 0.60), (400, 0.80), (1000, 1.00),
]

# --- Match engine -----------------------------------------------------------------
# Per-hole scoring distribution for an "average PGA Tour field" player.
# Mean relative to par: -(2*0.012 + 0.230) + 0.155 + 2*0.033 = -0.035 strokes,
# i.e. roughly a 70.4 scoring average on a par 72.
BASE_HOLE_DIST = {"eagle": 0.012, "birdie": 0.230, "par": 0.570, "bogey": 0.155, "dbl": 0.033}
LAMBDA_SENSITIVITY = 2.45  # dE[strokes vs par]/d-lambda per hole
FOURSOMES_PENALTY = 0.12    # strokes per round vs partners' average (alternate shot)
# Match play sharpness: amplifies per-hole quality gaps so that a 2.0
# strokes/round favorite wins ~77% of 18-hole singles matches, matching
# real-world match-play dynamics (calibrated Oct 2026; see README).
MATCH_SHARPNESS = 1.9
# European home advantage at Adare Manor (strokes per round), applied only in
# the full-cup simulation: home sides have won most modern Ryder Cups.
HOME_ADVANTAGE = 0.20

SIM_SEED = 20270917
MATCH_SIMS_MATRIX = 12000
MATCH_SIMS_DETAIL = 30000
SELECTION_SIMS = 20000
SESSION_SIMS = 10000
MAX_EXTRA_HOLES = 10

# --- Selection model ----------------------------------------------------------------
# Component weights early in a qualification race (form/rank driven) ramp toward
# official points as the race matures. This is what makes the model "change its
# mind" as 2027 results arrive.
W_EARLY = {"form": 1.00, "rank": 0.80, "points": 0.10, "experience": 0.25, "momentum": 0.30}
W_LATE = {"form": 0.55, "rank": 0.25, "points": 2.20, "experience": 0.25, "momentum": 0.15}
SELECTION_NOISE = 0.30      # per-draw noise on selection scores
POINTS_NOISE_FRACTION = 0.07  # auto-spot uncertainty, shrinks as the race matures
PICK_EXPERIENCE_BONUS = 0.15
MANUAL_EVENT_WEIGHT = 0.85  # confidence in hand-entered results vs feed results

# --- Team eligibility -----------------------------------------------------------------
EUROPE_COUNTRIES = {
    "England", "Scotland", "Wales", "Northern Ireland", "Ireland",
    "Spain", "Portugal", "France", "Belgium", "Netherlands", "Luxembourg",
    "Germany", "Austria", "Switzerland", "Italy", "Denmark", "Sweden", "Norway",
    "Finland", "Iceland", "Poland", "Czech Republic", "Czechia", "Hungary",
    "Slovakia", "Slovenia", "Croatia", "Serbia", "Greece", "Malta", "Cyprus",
    "Estonia", "Latvia", "Lithuania", "Monaco", "Andorra", "Gibraltar",
}
# Players from outside these countries (e.g. Ryan Fox, NZL) can still qualify for
# Europe as DP World Tour members; they are handled by explicit seed-pool entries.

TOP_DISCOVERY_RANK = 100  # auto-add OWGR top-100 Americans/Europeans missing from the seed pool
