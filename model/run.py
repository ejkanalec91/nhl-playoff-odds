"""Nightly job: fetch results, refit ratings, simulate, write the site's JSON.

Usage:
    python -m model.run              # update for today (and backfill missing days)
    python -m model.run --rebuild    # recompute every day of the season from scratch
    python -m model.run --sims 2000  # fewer simulations (faster, noisier)
"""

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from . import nhl_api, ratings as rt
from .simulate import simulate

DATA_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
MODEL_VERSION = "v1"


def today_eastern():
    return datetime.now(ZoneInfo("America/New_York")).date()


def load_json(name, default):
    path = DATA_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(name, data):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def standings(done, teams):
    """Current totals from finished games."""
    rec = {t: dict(gp=0, w=0, l=0, otl=0, pts=0, rw=0, row=0, gf=0, ga=0) for t in teams}
    for g in done:
        h, a = rec[g["home"]], rec[g["away"]]
        home_won = g["home_goals"] > g["away_goals"]
        win, lose = (h, a) if home_won else (a, h)
        win["w"] += 1
        win["pts"] += 2
        if g["last_period"] == "REG":
            win["rw"] += 1
            lose["l"] += 1
        else:
            lose["otl"] += 1
            lose["pts"] += 1
        if g["last_period"] != "SO":
            win["row"] += 1
        for side, gf, ga in ((h, g["home_goals"], g["away_goals"]), (a, g["away_goals"], g["home_goals"])):
            side["gp"] += 1
            side["gf"] += gf
            side["ga"] += ga
    return rec


def run_as_of(as_of, games, teams, meta, params, prior, table, n_sims, with_stakes=False):
    """Fit and simulate using only games finished before `as_of`.

    with_stakes runs a second simulation with ratings held fixed, so a game's
    "stakes" measure the effect of the result itself rather than what a win
    implies about how good the team is.
    """
    cutoff = as_of.isoformat()
    done = [g for g in games if g["final"] and g["date"] < cutoff]
    remaining = [g for g in games if not (g["final"] and g["date"] < cutoff)]
    b2b = rt.back_to_backs(games)

    r, r_sd, *_ = rt.fit(done, teams, b2b, prior=prior, params=params)
    rec = standings(done, teams)
    idx = {t: i for i, t in enumerate(teams)}

    state = {
        "teams": teams,
        "conf": [meta[t]["conf"] for t in teams],
        "div": [meta[t]["div"] for t in teams],
        "pts": np.array([rec[t]["pts"] for t in teams], float),
        "rw": np.array([rec[t]["rw"] for t in teams], float),
        "row": np.array([rec[t]["row"] for t in teams], float),
        "wins": np.array([rec[t]["w"] for t in teams], float),
        "home": np.array([idx[g["home"]] for g in remaining], int),
        "away": np.array([idx[g["away"]] for g in remaining], int),
        "b2b": np.array([int(b2b[g["id"]][1]) - int(b2b[g["id"]][0]) for g in remaining], float),
    }
    seed = int(as_of.strftime("%Y%m%d"))
    sim = simulate(state, r, r_sd, params, table, n_sims, seed)
    stakes = simulate(state, r, np.zeros_like(r_sd), params, table, n_sims, seed + 1) if with_stakes else None

    margin = r[state["home"]] - r[state["away"]] + params.home_adv + params.b2b_penalty * state["b2b"]
    p_reg, p_tie, p_ot = table.probs(margin)
    game_probs = {}
    for k, g in enumerate(remaining):
        game_probs[g["id"]] = {
            "p_home": float(p_reg[k] + p_tie[k] * p_ot[k]),
            "p_ot": float(p_tie[k]),
            "b2b_home": b2b[g["id"]][0],
            "b2b_away": b2b[g["id"]][1],
        }
        if stakes:
            for key in ("home_if_win", "home_if_loss", "away_if_win", "away_if_loss"):
                game_probs[g["id"]][key] = float(stakes[key][k])
    return r, rec, sim, game_probs, remaining


def round_(x, n=4):
    return round(float(x), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--rebuild", action="store_true", help="recompute all history")
    ap.add_argument("--as-of", help="YYYY-MM-DD, defaults to today (US Eastern)")
    args = ap.parse_args()

    today = date.fromisoformat(args.as_of) if args.as_of else today_eastern()

    print("Fetching teams and schedules...")
    season, meta = nhl_api.current_teams()
    teams = sorted(meta)
    games = nhl_api.season_games(season, teams)

    prev_season = season - 10001
    prev_games = nhl_api.season_games(prev_season, teams, use_cache=True)
    prev_teams = sorted({g["home"] for g in prev_games} | {g["away"] for g in prev_games})
    params, prev_ratings = rt.league_params(prev_games, prev_teams)
    print(f"League params from {prev_season}: {params}")

    prior = np.array([rt.PRIOR_CARRYOVER * prev_ratings.get(t, 0.0) for t in teams])
    prior -= prior.mean()
    table = rt.OutcomeTable(params)

    saved = load_json("history.json", {})
    fresh = args.rebuild or saved.get("season") != season  # new season starts a new history
    history = {} if fresh else saved.get("days", {})
    pregame = {} if fresh else load_json("pregame.json", {})

    # Every day from opening night through today that we haven't computed yet;
    # today is always recomputed.
    opening = date.fromisoformat(games[0]["date"])
    days = []
    d = opening
    while d <= today:
        if d.isoformat() not in history or d == today:
            days.append(d)
        d += timedelta(days=1)
    if not days:
        days = [today]

    for d in days:
        r, rec, sim, game_probs, remaining = run_as_of(
            d, games, teams, meta, params, prior, table, args.sims, with_stakes=(d == days[-1]))
        history[d.isoformat()] = {
            t: [round_(sim["playoffs"][i], 3), round_(sim["cup"][i], 3), round_(sim["proj_pts"][i], 1)]
            for i, t in enumerate(teams)
        }
        for g in remaining:
            if g["date"] == d.isoformat():
                pregame[str(g["id"])] = round_(game_probs[g["id"]]["p_home"], 3)
        print(f"  {d}: {sum(1 for g in games if g['final'] and g['date'] < d.isoformat())} games final")

    # Detailed output for the latest day
    latest_day = max(days)
    sos = {}
    for i, t in enumerate(teams):
        opp = [g["away"] if g["home"] == t else g["home"] for g in remaining if t in (g["home"], g["away"])]
        sos[t] = float(np.mean([r[teams.index(o)] for o in opp])) if opp else 0.0
    sos_rank = {t: k + 1 for k, t in enumerate(sorted(teams, key=lambda t: -sos[t]))}

    team_rows = []
    for i, t in enumerate(teams):
        team_rows.append({
            "abbrev": t, **meta[t], **{k: v for k, v in rec[t].items()},
            "rating": round_(r[i], 3),
            "proj_pts": round_(sim["proj_pts"][i], 1),
            "pts_p10": round_(sim["pts_p10"][i], 0),
            "pts_p90": round_(sim["pts_p90"][i], 0),
            "sos": round_(sos[t], 3),
            "sos_rank": sos_rank[t],
            **{k: round_(sim[k][i]) for k in
               ("playoffs", "div_win", "wildcard", "presidents", "round2", "conf_final", "final", "cup")},
        })

    write_json("latest.json", {
        "as_of": latest_day.isoformat(),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "sims": args.sims,
        "model": MODEL_VERSION,
        "params": {k: round_(v) for k, v in params.__dict__.items()},
        "teams": team_rows,
    })

    game_rows = []
    for g in games:
        row = {"id": g["id"], "date": g["date"], "home": g["home"], "away": g["away"]}
        if g["final"]:
            row.update(hg=g["home_goals"], ag=g["away_goals"], lp=g["last_period"])
        if str(g["id"]) in pregame:
            row["pre"] = pregame[str(g["id"])]
        if g["id"] in game_probs:
            p = game_probs[g["id"]]
            row.update({k: (round_(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
        game_rows.append(row)
    write_json("games.json", {"as_of": latest_day.isoformat(), "games": game_rows})

    write_json("history.json", {"season": season, "days": dict(sorted(history.items()))})
    write_json("pregame.json", dict(sorted(pregame.items())))
    print(f"Done. Wrote {DATA_DIR}")


if __name__ == "__main__":
    main()
