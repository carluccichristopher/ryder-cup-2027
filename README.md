# Ryder Cup 2027 Predictor ⛳

A self-updating Monte Carlo model for the **46th Ryder Cup** at **Adare Manor, County Limerick, Ireland (17–19 September 2027)**.

It answers three questions and keeps changing its mind as the 2027 season unfolds:

1. **Who makes each 12-man team?** Probabilities for every candidate, mirroring the real qualification rules.
2. **What is the optimal matchup for each player?** The full 12×12 singles matrix, solved for each captain's ideal board.
3. **How likely is each result — down to the exact scoreline?** Not just "61% to win", but `3&2`, `2&1`, `1UP`, or a 19th-hole finish.

Live dashboard: **GitHub Pages** (see the repository's Pages settings for the URL).

---

## How it works

| Stage | What happens |
| --- | --- |
| **Data** | `model/fetch_espn.py` pulls the Official World Golf Ranking and every completed PGA Tour leaderboard from ESPN's public JSON feeds, with a disk cache and stale-cache fallback so a network blip never breaks a run. DP World Tour results and official European points snapshots live in hand-maintained files under `data/manual/`. |
| **Ratings** | `model/ratings.py` scores each player-week in strokes per round versus that field's average, weighted by event strength, recency (26-week half-life) and rounds played, then shrinks toward an OWGR-derived prior. One number per player: strokes per round vs an average PGA Tour field. |
| **Team prediction** | `model/team_model.py` blends form, world rank, qualifying points, Ryder Cup experience and recent momentum into a selection score. Early in the race form and rank dominate; as 2027 results land, weight flows to the official points races. A 20,000-draw Monte Carlo of the real process (6 automatic qualifiers, then 6 captain's picks) produces P(auto), P(pick) and P(team). |
| **Match engine** | `model/matchplay.py` turns each rating into a per-hole scoring distribution (eagle/birdie/par/bogey/double) and simulates matches hole by hole, producing win probabilities **and exact scoreline distributions**. Supports singles, four-ball (best ball) and foursomes (alternate shot with a rhythm penalty). |
| **Lineups** | `model/optimizer.py` solves the 12×12 expected-points matrix with the Hungarian algorithm: USA's ideal board, Europe's ideal board, and every player's dream and nightmare opponent. It also ranks four-ball and foursomes pairings. |
| **The Cup** | 10,000 full 28-point Ryder Cups: Friday and Saturday each play 4 four-ball + 4 foursomes with the suggested pairings, then 12 singles, with a small home edge for Europe. Output includes P(USA win), P(Europe win), P(14–14) and the full margin distribution. |

Everything lands in `docs/data/model_output.json`, which the static dashboard in `docs/` reads. The dashboard's Match Lab re-implements the same engine in JavaScript, so you can simulate any pairing live in the browser and the numbers agree with the precomputed matrix.

### Calibration

- **Match sharpness `1.9`** — scales per-hole quality gaps so a 2-stroke-per-round favourite wins about 77% of 18-hole singles, matching real match play. Without it, hole-by-hole noise compresses everything toward a coin flip.
- **Home advantage `0.20` strokes/round** — applied to Europe in the full-cup simulation only; home sides have won most modern Ryder Cups.
- The assignment solver is verified against brute force on 200 random matrices (`python3 -m model.optimizer`), and the match engine has sanity tests (`python3 -m model.matchplay`).

---

## The 2027 qualification rules (as announced)

**Team USA** (captain Jim Furyk; announced 9 September 2026) — points from **2027 events only**, 31 of them, from The American Express in January through the BMW Championship (ends 22 August 2027). Winner points: majors 3,000; THE PLAYERS 2,850; Signature and Playoff events 2,250; full-field events 1,500. Opposite-field events do not count. Top 6 qualify automatically; Furyk picks six more the week after the TOUR Championship.

**Team Europe** (captain Luke Donald; announced 4 August 2026) — a single points list running **27 August 2026 → 22 August 2027** for eligible DP World Tour members. Event pools shared by finishing position: majors 5,000; PGA TOUR Signature/Players/Playoffs 3,000; Rolex Series and regular PGA TOUR 2,000; DP World Tour Back 9 1,500; Global Series 1,000. Leading six qualify automatically; Donald picks six more.

Because the American race does not start until January 2027, the model leans on form and world ranking for Team USA today and shifts toward hard points as the season progresses. That transition is the whole point: **the model's opinion changes with the season, in public, with a git history of every revision.**

---

## Running it yourself

```bash
git clone <your-repo-url>
cd ryder-cup-2027
pip3 install -r requirements.txt          # numpy only

python3 -m model.pipeline                 # fetch data + run the model (~1-2 min first time)
cd docs && python3 -m http.server 8642    # then open http://localhost:8642
```

Self-tests:

```bash
python3 -m model.optimizer    # Hungarian algorithm vs brute force
python3 -m model.matchplay    # match engine sanity checks
```

### Knobs worth turning

| File | What you can change |
| --- | --- |
| `model/config.py` | Match sharpness, home advantage, form half-life, selection weights (`W_EARLY` → `W_LATE`), simulation counts, event classifications, points schedules. |
| `data/seed_pool.json` | The candidate pool: who is eligible for which side, Ryder Cup appearances, LIV flags, manual rating overrides. |
| `data/manual/europe_points.json` | The official European points snapshot. Update it when Ryder Cup Europe publishes a new list; these values override modeled points. |
| `data/manual/results_log.json` | Hand-entered results for events the feeds miss (DP World Tour, LIV). Each finish becomes a rating observation and estimated points. |
| `data/manual/overrides.json` | Per-player eligibility, notes, or ratings. |

---

## Automation

`.github/workflows/update.yml` runs on **Mondays and Thursdays at 12:00 UTC** (and on demand via *Actions → Run workflow*). It installs numpy, runs the pipeline, and commits `docs/data/model_output.json` if anything changed. GitHub Pages serves `docs/` from the `main` branch, so a model refresh deploys itself.

After 2027 tournaments start finishing, each run will shuffle ratings, team probabilities, matchups and scoreline distributions. `git log docs/data/model_output.json` becomes a week-by-week diary of the model reacting to the season.

---

## Project layout

```
ryder-cup-2027/
├── model/
│   ├── config.py        competition facts, points schedules, model constants
│   ├── fetch_espn.py    ESPN/OWGR data layer with caching + fallback
│   ├── pool.py          candidate pool: seed file + live ranking merge
│   ├── ratings.py       form ratings and qualification points
│   ├── team_model.py    selection scores and team probabilities
│   ├── matchplay.py     hole-by-hole Monte Carlo match engine
│   ├── optimizer.py     Hungarian assignment, pairings, full-cup simulation
│   └── pipeline.py      orchestrates everything → docs/data/model_output.json
├── data/
│   ├── seed_pool.json   curated candidates for both sides
│   └── manual/          official points snapshot, results log, overrides
├── docs/                the GitHub Pages dashboard (static HTML/CSS/JS)
│   ├── index.html
│   ├── styles.css
│   ├── app.js           rendering + a JS port of the match engine
│   └── data/model_output.json
└── .github/workflows/update.yml
```

---

## Caveats

A fan project, not affiliated with Ryder Cup Europe, the PGA of America or the DP World Tour. Estimated points approximate the published schedules rather than reproducing them exactly; where official snapshots exist they take precedence and are labelled "official" in the dashboard. Course fit at Adare Manor, captaincy psychology, injuries and the Irish crowd are not modelled beyond a single home-advantage term. Probabilities are model opinion, not betting advice.
