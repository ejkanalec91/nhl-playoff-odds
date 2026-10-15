"""Backtest: replay past seasons and grade the model against what really happened.

For every game day, ratings are refit using only games played before that day,
then that day's games are predicted. On the first of each month the full
season simulation also runs, and its playoff odds are graded at season's end.

Usage:
    python -m model.backtest          # grade the current settings
    python -m model.backtest --tune   # also search for better settings

Tuning only looks at the TRAIN seasons; the TEST season is held out so the
reported accuracy is honest (the model never "saw" it while being tuned).
"""

import argparse
import itertools
import json
from datetime import date

import numpy as np

from . import nhl_api, ratings as rt
from .run import DATA_DIR, split_games, sim_state
from .simulate import simulate

TRAIN = [20232024, 20242025]
TEST = [20252026]
ALIASES = {"UTA": "ARI"}  # Utah took over Arizona's roster in 2024
CHECKPOINTS = ["11-01", "12-01", "01-01", "02-01", "03-01", "04-01"]
PLAYOFF_SIMS = 5000

GRID = {
    "shrink": [15, 25, 35, 50, 75],
    "carryover": [0.3, 0.5, 0.7],
    "margin_cap": [3, 4, 5],
}


def load_season(season):
    """Everything needed to replay one season (all API calls are cached)."""
    meta = nhl_api.final_standings(season)
    prev_meta = nhl_api.final_standings(season - 10001)
    games = [g for g in nhl_api.season_games(season, meta, use_cache=True) if g["final"]]
    prev_games = nhl_api.season_games(season - 10001, prev_meta, use_cache=True)
    teams = sorted(meta)
    b2b = rt.back_to_backs(games)
    X, ctx, margin = rt.design(games, teams, b2b)
    return {
        "season": season, "meta": meta, "teams": teams, "games": games, "b2b": b2b,
        "prev_games": prev_games, "prev_teams": sorted(prev_meta),
        "X": X, "ctx": ctx, "margin": margin,
        "home_won": np.array([g["home_goals"] > g["away_goals"] for g in games], float),
        "dates": np.array([g["date"] for g in games]),
    }


def setup(sd, settings):
    """League constants, outcome table and preseason prior for a season."""
    params, prev_ratings = rt.league_params(sd["prev_games"], sd["prev_teams"], settings)
    prior = rt.preseason_prior(sd["teams"], prev_ratings, settings, ALIASES)
    return params, rt.OutcomeTable(params), prior


def predict_games(sd, settings):
    """Pregame home-win probability for every game, using only earlier games."""
    params, table, prior = setup(sd, settings)
    dates = sd["dates"]
    hc = np.array([params.home_adv, params.b2b_penalty])
    p = np.zeros(len(dates))
    for d in np.unique(dates):
        n_before = np.searchsorted(dates, d)       # games are sorted by date
        today = np.where(dates == d)[0]
        r = rt.solve(sd["X"][:n_before], sd["ctx"][:n_before], sd["margin"][:n_before],
                     prior, settings.shrink, settings.margin_cap, params)[0]
        p[today] = table.home_win(sd["X"][today] @ r + sd["ctx"][today] @ hc)
    return p


def score(p, y):
    """Brier score, log loss and pick accuracy (lower is better for the first two)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "n": int(len(y)),
    }


def calibration(p, y, width):
    """Bucket predictions; compare average prediction to how often it happened."""
    bins = []
    edges = np.arange(0, 1 + 1e-9, width)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.sum():
            bins.append({"lo": round(lo, 3), "hi": round(hi, 3), "n": int(mask.sum()),
                         "predicted": round(float(p[mask].mean()), 4),
                         "actual": round(float(y[mask].mean()), 4)})
    return bins


def playoff_checkpoints(sd, settings):
    """Simulate from the first of each month; return rows of (month, team, p_model, p_coinflip, made)."""
    params, table, prior = setup(sd, settings)
    teams, meta = sd["teams"], sd["meta"]
    start_year = sd["season"] // 10000
    rows = []
    for mmdd in CHECKPOINTS:
        year = start_year if mmdd >= "07-01" else start_year + 1
        as_of = date.fromisoformat(f"{year}-{mmdd}")
        done, remaining = split_games(sd["games"], as_of)
        r, r_sd, *_ = rt.fit(done, teams, sd["b2b"], prior=prior, params=params, settings=settings)
        _, state = sim_state(done, remaining, teams, meta, sd["b2b"])
        seed = int(as_of.strftime("%Y%m%d"))
        model = simulate(state, r, r_sd, params, table, PLAYOFF_SIMS, seed)["playoffs"]
        zeros = np.zeros(len(teams))
        coin = simulate(state, zeros, zeros, params, table, PLAYOFF_SIMS, seed)["playoffs"]
        for i, t in enumerate(teams):
            rows.append((mmdd, t, float(model[i]), float(coin[i]), float(meta[t]["made_playoffs"])))
    return rows


def tune(train, test):
    """Grid-search settings, choosing by game log loss on the TRAIN seasons.

    The test-season score is recorded for each combination only so we can see
    whether the winner holds up; it plays no part in the choice.
    """
    def log_loss(seasons, s):
        p = np.concatenate([predict_games(sd, s) for sd in seasons])
        y = np.concatenate([sd["home_won"] for sd in seasons])
        return score(p, y)["log_loss"]

    results = []
    for shrink, carry, cap in itertools.product(*GRID.values()):
        s = rt.Settings(shrink=shrink, carryover=carry, margin_cap=cap)
        results.append({"shrink": shrink, "carryover": carry, "margin_cap": cap,
                        "train_log_loss": log_loss(train, s), "test_log_loss": log_loss(test, s)})
    results.sort(key=lambda r: r["train_log_loss"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="search settings and report the best")
    ap.add_argument("--adopt", action="store_true", help="grade the tuned winner instead of the live settings")
    args = ap.parse_args()

    print("Loading seasons (first run downloads ~200 files, later runs use the cache)...")
    data = {s: load_season(s) for s in TRAIN + TEST}
    train = [data[s] for s in TRAIN]

    current = rt.DEFAULT
    tuning = None
    if args.tune:
        print(f"Tuning on {TRAIN} ({np.prod([len(v) for v in GRID.values()])} combinations)...")
        tuning = tune(train, [data[s] for s in TEST])
        best = tuning[0]
        print("Best on train:", best)
        if args.adopt:
            current = rt.Settings(shrink=best["shrink"], carryover=best["carryover"], margin_cap=best["margin_cap"])

    report = {"settings": current.__dict__, "train": TRAIN, "test": TEST, "seasons": {}, "tuning": tuning}

    all_p, all_y, test_p, test_y = [], [], [], []
    for season, sd in data.items():
        p = predict_games(sd, current)
        p_default = predict_games(sd, rt.DEFAULT)
        y = sd["home_won"]
        prev_params, _ = rt.league_params(sd["prev_games"], sd["prev_teams"])
        prev_done = [g for g in sd["prev_games"] if g["final"]]
        home_rate = np.mean([g["home_goals"] > g["away_goals"] for g in prev_done])
        report["seasons"][season] = {
            "model": score(p, y),
            "v1_defaults": score(p_default, y),
            "home_ice_only": score(np.full(len(y), home_rate), y),
            "coin_flip": score(np.full(len(y), 0.5), y),
        }
        all_p.append(p); all_y.append(y)
        if season in TEST:
            test_p.append(p); test_y.append(y)

    all_p, all_y = np.concatenate(all_p), np.concatenate(all_y)
    test_p, test_y = np.concatenate(test_p), np.concatenate(test_y)
    report["games_all"] = score(all_p, all_y)
    report["games_test"] = score(test_p, test_y)
    report["game_calibration"] = calibration(all_p, all_y, 0.05)

    print("Simulating playoff odds at monthly checkpoints...")
    rows = []
    for season, sd in data.items():
        rows += [(season, *r) for r in playoff_checkpoints(sd, current)]
    by_month = {}
    for mmdd in CHECKPOINTS:
        sel = [r for r in rows if r[1] == mmdd]
        y = np.array([r[5] for r in sel])
        by_month[mmdd] = {
            "model": score(np.array([r[3] for r in sel]), y),
            "coin_flip": score(np.array([r[4] for r in sel]), y),
        }
    report["playoffs_by_month"] = by_month
    pm = np.array([r[3] for r in rows]); py = np.array([r[5] for r in rows])
    report["playoff_calibration"] = calibration(pm, py, 0.1)
    report["playoffs_all"] = {"model": score(pm, py), "coin_flip": score(np.array([r[4] for r in rows]), py)}

    out = DATA_DIR / "accuracy.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("settings", "games_all", "games_test", "playoffs_all")}, indent=1))
    for season, s in report["seasons"].items():
        print(season, {k: round(v["log_loss"], 4) for k, v in s.items()}, "acc", round(s["model"]["accuracy"], 3))
    for m, s in by_month.items():
        print(m, "playoff brier model", round(s["model"]["brier"], 4), "coin", round(s["coin_flip"]["brier"], 4))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
