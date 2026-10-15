"""Team strength ratings and single-game outcome probabilities.

v1 model: each team has one number, its rating, measured in goals per game
above an average team. Ratings are fit by ridge regression on goal margins,
shrunk toward a preseason prior (half of last season's rating).
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

MARGIN_CAP = 4          # cap blowouts (empty-net goals add noise, not signal)
IN_SEASON_SHRINK = 35   # prior weight, in "games", for in-season ratings
PRIOR_CARRYOVER = 0.5   # share of last season's rating carried into this season


@dataclass
class LeagueParams:
    """League-wide constants, fit from last season."""
    home_adv: float      # goals of margin from playing at home
    b2b_penalty: float   # goals of margin lost on the second night of a back-to-back
    sigma: float         # std. dev. of single-game margin around expectation
    base_goals: float    # average goals per team per game (excluding shootouts)
    tie_rate: float      # share of games tied after regulation
    so_share: float      # share of tied games decided by shootout


def game_margin(g):
    """Home goal margin, treating shootout games as ties."""
    if g["last_period"] == "SO":
        return 0.0
    return float(np.clip(g["home_goals"] - g["away_goals"], -MARGIN_CAP, MARGIN_CAP))


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


def fit(games, teams, b2b, prior=None, shrink=IN_SEASON_SHRINK, params=None):
    """Ridge-regress goal margins on team ratings.

    If params is None, home advantage and back-to-back effect are also fit
    (used on last season's data). Otherwise they are held fixed.
    Returns (ratings, rating_sd, home_adv, b2b_penalty, sigma).
    """
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    prior = np.zeros(n_teams) if prior is None else prior
    fit_context = params is None
    n_cols = n_teams + (2 if fit_context else 0)

    X = np.zeros((len(games), n_cols))
    y = np.zeros(len(games))
    for row, g in enumerate(games):
        X[row, idx[g["home"]]] = 1
        X[row, idx[g["away"]]] = -1
        hb, ab = b2b[g["id"]]
        y[row] = game_margin(g)
        if fit_context:
            X[row, n_teams] = 1
            X[row, n_teams + 1] = int(ab) - int(hb)
        else:
            y[row] -= params.home_adv + params.b2b_penalty * (int(ab) - int(hb))

    # Penalize team ratings toward the prior; the context terms are unpenalized
    penalty = np.zeros(n_cols)
    penalty[:n_teams] = shrink
    target = np.zeros(n_cols)
    target[:n_teams] = prior

    A = X.T @ X + np.diag(penalty)
    if fit_context:
        A[n_teams:, n_teams:] += np.eye(2) * 1e-6
    b = X.T @ y + penalty * target
    coef = np.linalg.solve(A, b)

    resid = y - X @ coef
    sigma = float(np.sqrt(np.mean(resid ** 2))) if len(games) > 50 else (params.sigma if params else 2.4)
    rating_sd = sigma * np.sqrt(np.diag(np.linalg.inv(A))[:n_teams])

    if fit_context:
        return coef[:n_teams], rating_sd, float(coef[n_teams]), float(coef[n_teams + 1]), sigma
    return coef[:n_teams], rating_sd, params.home_adv, params.b2b_penalty, sigma


def league_params(prev_games, prev_teams):
    """Fit league constants and last season's ratings from a finished season."""
    done = [g for g in prev_games if g["final"]]
    b2b = back_to_backs(prev_games)
    ratings, _, home_adv, b2b_pen, sigma = fit(done, prev_teams, b2b, shrink=2)

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
