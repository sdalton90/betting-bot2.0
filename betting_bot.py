import csv
import os
import sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SPORT = os.getenv("SPORT", "NBA").upper()
DB_FILE = os.getenv("DB_FILE", f"{SPORT.lower()}_picks.sqlite")
CSV_FILE = os.getenv("CSV_FILE", f"{SPORT.lower()}_picks_log.csv")

# Risk & edge rules (zero-cost conservative defaults)
BANKROLL_START = 1000.0
MIN_EDGE = 0.03          # Require at least 3.0% edge over no-vig market probability
KELLY_FRACTION = 0.25    # Quarter Kelly for lower volatility
MAX_BET_PCT = 0.03       # Never wager more than 3% of bankroll on one pick
MAX_STALENESS_MIN = 120  # Reject feeds older than 2 hours

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        game_date TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_score INTEGER,
        away_score INTEGER,
        status TEXT NOT NULL,
        lineup_confirmed INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        pick_side TEXT NOT NULL,
        team TEXT NOT NULL,
        odds_decimal REAL NOT NULL,
        model_prob REAL NOT NULL,
        no_vig_prob REAL NOT NULL,
        edge REAL NOT NULL,
        stake REAL NOT NULL,
        closing_odds REAL,
        clv_pct REAL,
        result TEXT,
        pnl REAL,
        created_at TEXT NOT NULL
    );
    """)
    conn.commit()

def sample_historical_games() -> pd.DataFrame:
    """Starter sample history so the script runs immediately on day 1."""
    rows = [
        ("G1", "2026-10-22", "Celtics", "Knicks", 132, 109, "final", 1),
        ("G2", "2026-10-23", "Nuggets", "Thunder", 87, 102, "final", 1),
        ("G3", "2026-10-25", "Knicks", "Pacers", 123, 98, "final", 1),
        ("G4", "2026-10-26", "Celtics", "Wizards", 122, 102, "final", 1),
        ("G5", "2026-10-28", "Thunder", "Hawks", 128, 104, "final", 1),
        ("G6", "2026-10-29", "Nuggets", "Nets", 144, 139, "final", 1),
        ("G7", "2026-10-30", "Knicks", "Heat", 116, 107, "final", 1),
        ("G8", "2026-11-01", "Celtics", "Hornets", 124, 109, "final", 1),
    ]
    return pd.DataFrame(
        rows,
        columns=["game_id", "game_date", "home_team", "away_team", "home_score", "away_score", "status", "lineup_confirmed"],
    )

def build_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True).copy()
    elo: dict[str, float] = {}
    margins: dict[str, list[float]] = {}
    pre_home_elo, pre_away_elo, pre_diff = [], [], []

    for r in df.itertuples(index=False):
        h_elo = elo.get(r.home_team, 1500.0)
        a_elo = elo.get(r.away_team, 1500.0)
        pre_home_elo.append(h_elo)
        pre_away_elo.append(a_elo)

        h_m = margins.get(r.home_team, [])[-window:]
        a_m = margins.get(r.away_team, [])[-window:]
        h_avg = float(np.mean(h_m)) if h_m else 0.0
        a_avg = float(np.mean(a_m)) if a_m else 0.0
        pre_diff.append(h_avg - a_avg)

        if r.status == "final" and r.home_score is not None and r.away_score is not None:
            actual = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
            exp_home = 1.0 / (1.0 + 10.0 ** (((a_elo) - (h_elo + 45.0)) / 400.0))
            delta = 20.0 * (actual - exp_home)
            elo[r.home_team] = h_elo + delta
            elo[r.away_team] = a_elo - delta

            diff = float(r.home_score - r.away_score)
            margins.setdefault(r.home_team, []).append(diff)
            margins.setdefault(r.away_team, []).append(-diff)

    df["home_elo"] = pre_home_elo
    df["away_elo"] = pre_away_elo
    df["elo_gap"] = (df["home_elo"] + 45.0) - df["away_elo"]
    df["margin_gap"] = pre_diff
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df

def no_vig_probs(home_odds: float, away_odds: float) -> tuple[float, float]:
    p_h = 1.0 / home_odds
    p_a = 1.0 / away_odds
    total = p_h + p_a
    return p_h / total, p_a / total

def kelly_stake(bankroll: float, prob: float, odds: float, edge: float) -> float:
    if edge < MIN_EDGE or odds <= 1.0 or bankroll <= 0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    full_k = ((b * prob) - q) / b
    if full_k <= 0:
        return 0.0
    frac = min(full_k * KELLY_FRACTION, MAX_BET_PCT)
    return round(bankroll * frac, 2)

def export_csv(conn: sqlite3.Connection) -> None:
    df = pd.read_sql_query("SELECT * FROM picks ORDER BY id DESC", conn)
    df.to_csv(CSV_FILE, index=False)

def main() -> None:
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    hist = sample_historical_games()
    feat = build_rolling_features(hist)

    clf = None
    if len(feat) >= 6 and feat["home_win"].nunique() > 1:
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=80, random_state=42)
        clf.fit(feat[["elo_gap", "margin_gap"]], feat["home_win"])

    # Example upcoming game (replace with live feed/API output)
    upcoming = {
        "game_id": "G_UPCOMING_1",
        "game_date": utc_now().strftime("%Y-%m-%d"),
        "home_team": "Knicks",
        "away_team": "Celtics",
        "lineup_confirmed": 1,
        "updated_at": utc_now().isoformat(),
        "home_odds": 2.25,
        "away_odds": 1.72,
    }

    age_min = (utc_now() - datetime.fromisoformat(upcoming["updated_at"])).total_seconds() / 60.0
    if not upcoming["lineup_confirmed"] or age_min > MAX_STALENESS_MIN:
        print("Skipped game: unconfirmed lineup or stale data.")
        conn.close()
        return

    h_elo = float(feat.loc[feat["home_team"] == upcoming["home_team"], "home_elo"].tail(1).squeeze()) if upcoming["home_team"] in set(feat["home_team"]) else 1500.0
    a_elo = float(feat.loc[feat["away_team"] == upcoming["away_team"], "away_elo"].tail(1).squeeze()) if upcoming["away_team"] in set(feat["away_team"]) else 1500.0
    elo_prob = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + 45.0)) / 400.0))

    if clf is not None:
        x_next = pd.DataFrame([{"elo_gap": (h_elo + 45.0) - a_elo, "margin_gap": 0.0}])
        model_home_prob = 0.5 * elo_prob + 0.5 * float(clf.predict_proba(x_next)[0, 1])
    else:
        model_home_prob = elo_prob

    nv_home, nv_away = no_vig_probs(upcoming["home_odds"], upcoming["away_odds"])
    model_away_prob = 1.0 - model_home_prob

    pnl_row = conn.execute("SELECT COALESCE(SUM(pnl), 0.0) FROM picks WHERE pnl IS NOT NULL").fetchone()
    bankroll = BANKROLL_START + float(pnl_row[0])

    candidates = [
        ("HOME", upcoming["home_team"], upcoming["home_odds"], model_home_prob, nv_home),
        ("AWAY", upcoming["away_team"], upcoming["away_odds"], model_away_prob, nv_away),
    ]

    for side, team, odds, m_prob, nv_prob in candidates:
        edge = m_prob - nv_prob
        stake = kelly_stake(bankroll, m_prob, odds, edge)
        if stake > 0:
            exists = conn.execute(
                "SELECT 1 FROM picks WHERE game_id = ? AND pick_side = ?",
                (upcoming["game_id"], side),
            ).fetchone()
            if not exists:
                conn.execute(
                    """
                    INSERT INTO picks (game_id, pick_side, team, odds_decimal, model_prob, no_vig_prob, edge, stake, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (upcoming["game_id"], side, team, odds, round(m_prob, 4), round(nv_prob, 4), round(edge, 4), stake, utc_now().isoformat()),
                )
                print(f"Logged +EV pick: {team} ({side}) @ {odds} | Edge: {edge:.2%} | Stake: ${stake:.2f}")

    conn.commit()
    export_csv(conn)
    conn.close()

if __name__ == "__main__":
    main()