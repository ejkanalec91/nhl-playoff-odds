"""Thin client for the public NHL web API (api-web.nhle.com)."""

import json
import time
from pathlib import Path

import requests

BASE = "https://api-web.nhle.com/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

FINAL_STATES = {"OFF", "FINAL"}


def get_json(path, cache_name=None, retries=4):
    """GET a JSON endpoint. If cache_name is given, reuse a cached copy on disk."""
    cache_file = CACHE_DIR / cache_name if cache_name else None
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE}/{path}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def current_teams():
    """Return (season_id, {abbrev: {name, short, conf, div}}) from current standings."""
    data = get_json("standings/now")
    teams = {}
    season = None
    for row in data["standings"]:
        abbrev = row["teamAbbrev"]["default"]
        season = row["seasonId"]
        teams[abbrev] = {
            "name": row["teamName"]["default"],
            "short": row["teamCommonName"]["default"],
            "conf": row["conferenceName"],
            "div": row["divisionName"],
        }
    return season, teams


def season_games(season, team_abbrevs, use_cache=False):
    """All regular-season games for a season, de-duplicated and normalized.

    Each game: {id, date, home, away, final, home_goals, away_goals, last_period}
    last_period is "REG", "OT" or "SO" for finished games, otherwise None.
    """
    games = {}
    for abbrev in sorted(team_abbrevs):
        cache_name = f"schedule_{season}_{abbrev}.json" if use_cache else None
        data = get_json(f"club-schedule-season/{abbrev}/{season}", cache_name)
        for g in data.get("games", []):
            if g["gameType"] != 2 or g["id"] in games:
                continue
            final = g["gameState"] in FINAL_STATES
            games[g["id"]] = {
                "id": g["id"],
                "date": g["gameDate"],
                "home": g["homeTeam"]["abbrev"],
                "away": g["awayTeam"]["abbrev"],
                "final": final,
                "home_goals": g["homeTeam"].get("score") if final else None,
                "away_goals": g["awayTeam"].get("score") if final else None,
                "last_period": g.get("gameOutcome", {}).get("lastPeriodType") if final else None,
            }
    return sorted(games.values(), key=lambda g: (g["date"], g["id"]))
