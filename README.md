# NHL Playoff Odds

Every NHL team's chance of making the playoffs and winning the Stanley Cup, plus win probabilities and stakes for every game. The odds update nightly from 10,000 simulations of the rest of the season.

## How it works

1. **Ratings** ([model/ratings.py](model/ratings.py)): fit each team's strength (goals per game vs. average) from goal margins, adjusted for opponent, home ice and back-to-backs, and shrunk toward a preseason prior based on last season.
2. **Simulation** ([model/simulate.py](model/simulate.py)): play out every remaining game (regulation/OT/shootout, loser points, NHL tiebreakers), seed the playoffs (top 3 per division plus 2 wildcards per conference), and simulate best-of-seven series through the Cup Final.
3. **Output** ([model/run.py](model/run.py)): write JSON to `site/data/`. The static site in `site/` reads it.

## Run it locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m model.run            # update today (backfills missing days)
.\.venv\Scripts\python -m http.server 8000 -d site
```

Then open http://localhost:8000.

Options: `--rebuild` recomputes the whole season's history, `--sims 2000` runs faster, `--as-of 2026-11-01` computes the odds as of a past date.

## Deploy

A GitHub Actions workflow ([.github/workflows/nightly.yml](.github/workflows/nightly.yml)) runs every morning. It reruns the model, commits the new data and publishes `site/` to GitHub Pages.

## Roadmap

- v2: expected goals (xG) model built from NHL play-by-play
- v3: starting-goalie adjustments

Unofficial fan project, not affiliated with the NHL.
