"""Monte Carlo simulation of the rest of the regular season and the playoffs."""

import numpy as np

BATCH = 2500  # sims per batch, keeps memory use modest


def _rank_keys(pts, rw, row, wins, rng):
    """One sortable number per team per sim, following NHL tiebreakers:
    points, regulation wins, regulation+OT wins, total wins, then random."""
    noise = rng.random(pts.shape) * 0.5
    return pts * 1e9 + rw * 1e6 + row * 1e3 + wins + noise


def _series(hi, lo, R, table, params, rng):
    """Simulate best-of-7 series; hi has home ice. Returns winners array."""
    sims = np.arange(len(hi))
    m_hi_home = R[sims, hi] - R[sims, lo] + params.home_adv
    m_lo_home = R[sims, lo] - R[sims, hi] + params.home_adv
    p_hi_home = table.home_win(m_hi_home)
    p_hi_away = 1 - table.home_win(m_lo_home)
    # 2-2-1-1-1 format: hi hosts games 1, 2, 5, 7. Playing all seven games and
    # counting wins gives the same winner as stopping at four.
    p = np.stack([p_hi_home, p_hi_home, p_hi_away, p_hi_away, p_hi_home, p_hi_away, p_hi_home], axis=1)
    hi_wins = (rng.random(p.shape) < p).sum(axis=1)
    return np.where(hi_wins >= 4, hi, lo)


def _pick(keys, a, b):
    """Return (higher seed, lower seed) between team arrays a and b."""
    sims = np.arange(len(a))
    a_better = keys[sims, a] > keys[sims, b]
    return np.where(a_better, a, b), np.where(a_better, b, a)


def simulate(state, ratings, rating_sd, params, table, n_sims, seed):
    """Run the simulation.

    state: dict with
      teams: list of abbrevs; conf, div: lists aligned with teams
      pts, rw, row, wins: current totals (arrays)
      home, away: remaining game team indices; b2b: away_b2b - home_b2b per game
    Returns a dict of per-team and per-game results.
    """
    rng = np.random.default_rng(seed)
    teams = state["teams"]
    n_teams = len(teams)
    home, away, b2b = state["home"], state["away"], state["b2b"]
    n_games = len(home)

    H = np.zeros((n_games, n_teams))
    A = np.zeros((n_games, n_teams))
    H[np.arange(n_games), home] = 1
    A[np.arange(n_games), away] = 1

    conf = np.array(state["conf"])
    div = np.array(state["div"])
    conferences = sorted(set(conf))
    divisions = {c: sorted(set(div[conf == c])) for c in conferences}

    totals = {k: np.zeros(n_teams) for k in
              ("playoffs", "div_win", "wildcard", "presidents", "round2", "conf_final", "final", "cup")}
    final_pts = []
    # Per-game, per-side leverage tallies: made playoffs when the side won / lost
    lev = {k: np.zeros(n_games) for k in ("hw", "h_made_hw", "a_made_hw", "h_made_hl", "a_made_hl")}

    done = 0
    while done < n_sims:
        S = min(BATCH, n_sims - done)
        done += S
        R = ratings + rating_sd * rng.standard_normal((S, n_teams))

        # Regular season
        if n_games:
            margin = R[:, home] - R[:, away] + params.home_adv + params.b2b_penalty * b2b
            p_reg, p_tie, p_ot = table.probs(margin)
            u = rng.random((S, n_games))
            home_reg = u < p_reg
            tied = (u >= p_reg) & (u < p_reg + p_tie)
            u2 = rng.random((S, n_games))
            home_ot = tied & (u2 < p_ot)
            away_ot = tied & ~home_ot
            shootout = tied & (rng.random((S, n_games)) < params.so_share)
            away_reg = ~home_reg & ~tied

            home_win = home_reg | home_ot
            away_win = away_reg | away_ot
            # 2 points for any win, 1 "loser point" for losing in OT/shootout
            pts = state["pts"] + (2 * home_win + away_ot) @ H + (2 * away_win + home_ot) @ A
            rw = state["rw"] + home_reg @ H + away_reg @ A
            row = state["row"] + (home_reg | (home_ot & ~shootout)) @ H + (away_reg | (away_ot & ~shootout)) @ A
            wins = state["wins"] + home_win @ H + away_win @ A
        else:
            home_win = np.zeros((S, 0), dtype=bool)
            pts, rw, row, wins = (np.tile(state[k], (S, 1)) for k in ("pts", "rw", "row", "wins"))

        keys = _rank_keys(pts, rw, row, wins, rng)
        final_pts.append(pts)
        sims = np.arange(S)

        made = np.zeros((S, n_teams), dtype=bool)
        top = keys.argmax(axis=1)
        totals["presidents"] += np.bincount(top, minlength=n_teams)

        conf_champs = []
        for c in conferences:
            seeds = {}
            in_div = set()
            for d in divisions[c]:
                members = np.where(div == d)[0]
                order = members[np.argsort(-keys[:, members], axis=1)]
                seeds[d] = order[:, :3]
                in_div.update(members.tolist())
                totals["div_win"] += np.bincount(order[:, 0], minlength=n_teams)
                for k in range(3):
                    made[sims, order[:, k]] = True

            conf_members = np.where(conf == c)[0]
            wc_keys = np.where(made[:, conf_members], -np.inf, keys[:, conf_members])
            wc_order = conf_members[np.argsort(-wc_keys, axis=1)]
            wc1, wc2 = wc_order[:, 0], wc_order[:, 1]
            made[sims, wc1] = True
            made[sims, wc2] = True
            totals["wildcard"] += np.bincount(wc1, minlength=n_teams) + np.bincount(wc2, minlength=n_teams)

            # Better division winner faces the lower wildcard
            d1, d2 = divisions[c]
            top1, top2 = seeds[d1][:, 0], seeds[d2][:, 0]
            d1_better = keys[sims, top1] > keys[sims, top2]
            opp1 = np.where(d1_better, wc2, wc1)
            opp2 = np.where(d1_better, wc1, wc2)

            bracket_winners = []
            for d, opp in ((d1, opp1), (d2, opp2)):
                w_a = _series(seeds[d][:, 0], opp, R, table, params, rng)
                hi, lo = _pick(keys, seeds[d][:, 1], seeds[d][:, 2])
                w_b = _series(hi, lo, R, table, params, rng)
                for w in (w_a, w_b):
                    totals["round2"] += np.bincount(w, minlength=n_teams)
                hi, lo = _pick(keys, w_a, w_b)
                w = _series(hi, lo, R, table, params, rng)
                totals["conf_final"] += np.bincount(w, minlength=n_teams)
                bracket_winners.append(w)

            hi, lo = _pick(keys, *bracket_winners)
            champ = _series(hi, lo, R, table, params, rng)
            totals["final"] += np.bincount(champ, minlength=n_teams)
            conf_champs.append(champ)

        hi, lo = _pick(keys, *conf_champs)
        totals["cup"] += np.bincount(_series(hi, lo, R, table, params, rng), minlength=n_teams)
        totals["playoffs"] += made.sum(axis=0)

        if n_games:
            h_made = made[:, home]
            a_made = made[:, away]
            lev["hw"] += home_win.sum(axis=0)
            lev["h_made_hw"] += (h_made & home_win).sum(axis=0)
            lev["a_made_hw"] += (a_made & home_win).sum(axis=0)
            lev["h_made_hl"] += (h_made & ~home_win).sum(axis=0)
            lev["a_made_hl"] += (a_made & ~home_win).sum(axis=0)

    final_pts = np.vstack(final_pts)
    result = {k: v / n_sims for k, v in totals.items()}
    result["proj_pts"] = final_pts.mean(axis=0)
    result["pts_p10"] = np.percentile(final_pts, 10, axis=0)
    result["pts_p90"] = np.percentile(final_pts, 90, axis=0)

    # Playoff odds for each side if it wins vs. loses the game
    hw = np.maximum(lev["hw"], 1)
    hl = np.maximum(n_sims - lev["hw"], 1)
    result["home_if_win"] = lev["h_made_hw"] / hw
    result["home_if_loss"] = lev["h_made_hl"] / hl
    result["away_if_win"] = lev["a_made_hl"] / hl
    result["away_if_loss"] = lev["a_made_hw"] / hw
    return result
