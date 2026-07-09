import pandas as pd
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text

connection_url = URL.create(
    drivername="postgresql",
    username="postgres",
    password="Infi@2k4",
    host="localhost",
    port=5432,
    database="fifa_analytics"
)
engine = create_engine(connection_url)

queries = {

# ── Q1: All-time win % per team (min 10 matches) ──────────────
"Q1_team_win_rates": """
    SELECT
        team,
        matches_played,
        wins,
        draws,
        losses,
        ROUND(wins * 100.0 / matches_played, 1) AS win_pct,
        ROUND((wins + draws * 0.5) * 100.0 / matches_played, 1) AS points_pct
    FROM (
        SELECT
            team,
            COUNT(*) AS matches_played,
            SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END) AS draws,
            SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses
        FROM (
            SELECT home_team AS team,
                   CASE WHEN result = 'home_win' THEN 'win'
                        WHEN result = 'draw'     THEN 'draw'
                        ELSE 'loss' END AS result
            FROM fact_matches
            UNION ALL
            SELECT away_team AS team,
                   CASE WHEN result = 'away_win' THEN 'win'
                        WHEN result = 'draw'     THEN 'draw'
                        ELSE 'loss' END AS result
            FROM fact_matches
        ) all_results
        GROUP BY team
    ) team_totals
    WHERE matches_played >= 10
    ORDER BY win_pct DESC
    LIMIT 20
""",

# ── Q2: Goals scored and conceded per team ────────────────────
"Q2_team_goals": """
    SELECT
        team,
        SUM(goals_for)     AS total_goals_for,
        SUM(goals_against) AS total_goals_against,
        SUM(goals_for) - SUM(goals_against) AS goal_difference,
        ROUND(AVG(goals_for), 2) AS avg_goals_per_match
    FROM (
        SELECT home_team AS team, home_score AS goals_for,  away_score AS goals_against FROM fact_matches
        UNION ALL
        SELECT away_team AS team, away_score AS goals_for,  home_score AS goals_against FROM fact_matches
    ) g
    GROUP BY team
    ORDER BY total_goals_for DESC
    LIMIT 20
""",

# ── Q3: Tournament summary stats ──────────────────────────────
"Q3_tournament_summary": """
    SELECT
        f.year,
        t.host,
        t.champion,
        COUNT(*)                        AS matches_played,
        SUM(f.total_goals)              AS total_goals,
        ROUND(AVG(f.total_goals), 2)    AS avg_goals_per_match,
        MAX(f.total_goals)              AS highest_scoring_match,
        ROUND(AVG(f.attendance), 0)     AS avg_attendance
    FROM fact_matches f
    JOIN dim_tournament t ON f.year = t.year
    GROUP BY f.year, t.host, t.champion
    ORDER BY f.year DESC
""",

# ── Q4: Host nation advantage ─────────────────────────────────
"Q4_host_advantage": """
    SELECT
        host_flag,
        COUNT(*)                                              AS matches,
        SUM(CASE WHEN result = 'home_win' THEN 1 ELSE 0 END) AS wins,
        SUM(CASE WHEN result = 'draw'     THEN 1 ELSE 0 END) AS draws,
        SUM(CASE WHEN result = 'away_win' THEN 1 ELSE 0 END) AS losses,
        ROUND(SUM(CASE WHEN result = 'home_win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS win_pct
    FROM (
        SELECT result,
               CASE WHEN home_team = host THEN 'Host nation'
                    ELSE 'Non-host' END AS host_flag
        FROM fact_matches
    ) h
    GROUP BY host_flag
""",

# ── Q5: Most prolific World Cup stages ────────────────────────
"Q5_goals_by_round": """
    SELECT
        round,
        COUNT(*)                        AS matches,
        SUM(total_goals)                AS total_goals,
        ROUND(AVG(total_goals), 2)      AS avg_goals,
        MAX(total_goals)                AS max_goals_in_match
    FROM fact_matches
    GROUP BY round
    ORDER BY avg_goals DESC
    LIMIT 15
""",

# ── Q6: Biggest upsets (lower ranked team won) ────────────────
"Q6_biggest_upsets": """
    SELECT
        f.year,
        f.round,
        f.home_team,
        f.away_team,
        f.home_score,
        f.away_score,
        f.result,
        r_home.rank AS home_rank,
        r_away.rank AS away_rank,
        ABS(r_home.rank - r_away.rank) AS rank_gap
    FROM fact_matches f
    LEFT JOIN dim_rankings r_home
           ON f.home_team = r_home.team_name
          AND r_home.snapshot_date = '2022-10-06'
    LEFT JOIN dim_rankings r_away
           ON f.away_team = r_away.team_name
          AND r_away.snapshot_date = '2022-10-06'
    WHERE
        (f.result = 'home_win' AND r_home.rank > r_away.rank)
     OR (f.result = 'away_win' AND r_away.rank > r_home.rank)
    ORDER BY rank_gap DESC
    LIMIT 10
""",

# ── Q7: Champions and their tournament stats ──────────────────
"Q7_champions": """
    SELECT
        t.champion,
        COUNT(*) AS titles,
        STRING_AGG(t.year::text, ', ' ORDER BY t.year) AS years_won
    FROM dim_tournament t
    GROUP BY t.champion
    ORDER BY titles DESC
"""
}

# ── RUN ALL QUERIES & SAVE RESULTS ───────────────────────────
print("Running SQL analytical queries...\n")
results = {}

for name, sql in queries.items():
    df = pd.read_sql(text(sql), engine)
    results[name] = df
    print(f"{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(df.to_string(index=False))
    print()

# Save all results to CSV
for name, df in results.items():
    df.to_csv(f'data/processed/{name}.csv', index=False)

print("✅ All queries complete. Results saved to data/processed/")