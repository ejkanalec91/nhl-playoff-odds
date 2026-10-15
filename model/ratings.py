"""Team strength ratings and single-game outcome probabilities.

v1 model: each team has one number, its rating, measured in goals per game
above an average team. Ratings are fit by ridge regression on goal margins,
shrunk toward a preseason prior (a share of last season's rating).
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass
class Settings:
    """Model knobs.

    Backtest (Oct 2026): tuning these on 2023-25 picked shrink=75, carryover=0.7,
    margin_cap=5, which scored worse on the held-out 2025-26 season, so the
    original values were kept. Differences across the grid were small (~0.003 log loss).
    """
    shrink: float = 35.0     # prior weight, in "games", for in-season ratings
    carryover: float = 0.5   # share of last season's rating carried into this season
    margin_cap: float = 4.0  # cap blowouts (empty-net goals add noise, not signal)


DEFAULT = Settings()


@dataclass
class LeagueParams:
    """League-wide constants, fit from last season."""
    home_adv: float      # goals of margin from playing at home
    b2b_penalty: float   # goals of margin lost on the second night of a back-to-back
    sigma: float         # std. dev. of single-game margin around expectation
    base_goals: float    # average goals per team per game (excluding shootouts)
    tie_rate: float      # share of games tied after regulation
    so_share: float      # share of tied games decided by shootout


def back_to_backs(games):
    """Map game id -> (home_on_b2b, away_on_b2b) using every scheduled game."""
    played = {}
    for g in games:
        for team in (g["home"], g["away"]):
            played.setdefault(team, set()).add(g["date"])
    result = {}
    for g in games:
        prev = (date.fromisoformat(g["date"]) - timedelta(days=1)).isoformat()
        result[g["id"]] = (prev in played[g["home"]], prev in played[g["away"]])
    return result


def design(games, teams, b2b):
    """Turn finished games into regression inputs.

    Returns (X, ctx, margin):
      X:      one row per game, +1 in the home team's column, -1 in the away team's
      ctx:    columns [1 (home ice), away_b2b - home_b2b]
      margin: home goals minus away goals, with shootout games counted as ties
    """
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(games), len(teams)))
    ctx = np.zeros((len(games), 2))
    margin = np.zeros(len(games))
    for row, g in enumerate(games):
        X[row, idx[g["home"]]] = 1
        X[row, idx[g["away"]]] = -1
        hb, ab = b2b[g["id"]]
        ctx[row] = (1, int(ab) - int(hb))
        margin[row] = 0 if g["last_period"] == "SO" else g["home_goals"] - g["away_goals"]
    return X, ctx, margin


def solve(X, ctx, margin, prior, shrink, cap, params=None):
    """Ridge-regress capped goal margins on team ratings.

    If params is None, home ice and the back-to-back effect are fit too (used on
    a finished season). Otherwise they are held at the given values.
    Returns (ratings, rating_sd, home_adv, b2b_penalty, sigma).
    """
    n_teams = X.shape[1]
    y = np.clip(margin, -cap, cap)
    if params is None:
        X = np.hstack([X, ctx])
    else:
        y = y - ctx @ np.array([params.home_adv, params.b2b_penalty])
    n_cols = X.shape[1]

    # Penalize team ratings toward the prior; the context terms are unpenalized
    penalty = np.zeros(n_cols)
    penalty[:n_teams] = shrink
    target = np.zeros(n_cols)
    target[:n_teams] = prior

    A = X.T @ X + np.diag(penalty)
    if params is None:
        A[n_teams:, n_teams:] += np.eye(2) * 1e-6
    coef = np.linalg.solve(A, X.T @ y + penalty * target)

    if len(y) > 50:
        sigma = float(np.sqrt(np.mean((y - X @ coef) ** 2)))
    else:
        sigma = params.sigma if params else 2.4
    rating_sd = sigma * np.sqrt(np.diag(np.linalg.inv(A))[:n_teams])

    if params is None:
        return coef[:n_teams], rating_sd, float(coef[n_teams]), float(coef[n_teams + 1]), sigma
    return coef[:n_teams], rating_sd, params.home_adv, params.b2b_penalty, sigma


def fit(games, teams, b2b, prior=None, settings=DEFAULT, params=None, shrink=None):
    """Convenience wrapper: design + solve."""
    prior = np.zeros(len(teams)) if prior is None else prior
    X, ctx, margin = design(games, teams, b2b)
    return solve(X, ctx, margin, prior, settings.shrink if shrink is None else shrink,
                 settings.margin_cap, params)


def league_params(prev_games, prev_teams, settings=DEFAULT):
    """Fit league constants and last season's ratings from a finished season."""
    done = [g for g in prev_games if g["final"]]
    b2b = back_to_backs(prev_games)
    ratings, _, home_adv, b2b_pen, sigma = fit(done, prev_teams, b2b, settings=settings, shrink=2)

    non_so_goals = [
        (g["home_goals"] + g["away_goals"] - (1 if g["last_period"] == "SO" else 0)) / 2
        for g in done
    ]
    extra = [g for g in done if g["last_period"] in ("OT", "SO")]
    params = LeagueParams(
        home_adv=home_adv,
        b2b_penalty=b2b_pen,
        sigma=sigma,
        base_goals=float(np.mean(non_so_goals)),
        tie_rate=len(extra) / len(done),
        so_share=sum(g["last_period"] == "SO" for g in extra) / max(len(extra), 1),
    )
    return params, dict(zip(prev_teams, ratings))


def preseason_prior(teams, prev_ratings, settings=DEFAULT, aliases=None):
    """Each team starts at a share of last season's rating (re-centered to zero).

    aliases maps a team to its previous-season abbreviation (e.g. relocations)."""
    aliases = aliases or {}
    prior = np.array([
        settings.carryover * prev_ratings.get(t, prev_ratings.get(aliases.get(t), 0.0)) for t in teams
    ])
    return prior - prior.mean()


class OutcomeTable:
    """Probability of each game result as a function of expected home margin.

    Margins follow a Skellam (difference of Poisson goal counts) shape, then
    the regulation-tie probability is rescaled to the league's real OT rate.
    """

    def __init__(self, params):
        self.grid = np.linspace(-5, 5, 2001)
        goals = np.arange(0, 21)
        log_fact = np.cumsum(np.log(np.maximum(goals, 1)))

        def pmf(lam):
            return np.exp(goals * np.log(lam) - lam - log_fact)

        win, tie = [], []
        for m in self.grid:
            ph = pmf(max(params.base_goals + m / 2, 0.3))
            pa = pmf(max(params.base_goals - m / 2, 0.3))
            joint = np.outer(ph, pa)
            win.append(np.tril(joint, -1).sum())
            tie.append(np.trace(joint))
        win, tie = np.array(win), np.array(tie)

        tie_at_even = np.interp(0, self.grid, tie)
        scaled_tie = np.clip(tie * params.tie_rate / tie_at_even, 0, 0.5)
        decisive = 1 - tie
        self.reg_home = win / decisive * (1 - scaled_tie)
        self.tie = scaled_tie
        # Better teams win OT/shootouts a bit more often than a coin flip
        self.ot_home = 1 / (1 + np.exp(-0.3 * self.grid))

    def probs(self, margin):
        """Return (p_home_regulation_win, p_tie_after_regulation, p_home_wins_ot)."""
        return (
            np.interp(margin, self.grid, self.reg_home),
            np.interp(margin, self.grid, self.tie),
            np.interp(margin, self.grid, self.ot_home),
        )

    def home_win(self, margin):
        reg, tie, ot = self.probs(margin)
        return reg + tie * ot
