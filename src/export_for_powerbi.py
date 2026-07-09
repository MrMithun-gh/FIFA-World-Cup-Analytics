import pandas as pd
from sqlalchemy.engine import URL
from sqlalchemy import create_engine

connection_url = URL.create(
    drivername="postgresql",
    username="postgres",
    password="Infi@2k4",
    host="localhost",
    port=5432,
    database="fifa_analytics"
)
engine = create_engine(connection_url)

print("Exporting tables for Power BI...")

# ─── LOAD ALL TABLES ─────────────────────────────────────────
matches    = pd.read_sql("SELECT * FROM fact_matches",   engine)
tournament = pd.read_sql("SELECT * FROM dim_tournament", engine)
teams      = pd.read_sql("SELECT * FROM dim_team",       engine)
rankings   = pd.read_sql("SELECT * FROM dim_rankings",   engine)
schedule   = pd.read_sql("SELECT * FROM dim_schedule",   engine)

# ─── CLEAN FOR POWER BI ──────────────────────────────────────
matches['total_goals'] = pd.to_numeric(matches['total_goals'], errors='coerce')
matches['attendance']  = pd.to_numeric(matches['attendance'],  errors='coerce')
matches['home_score']  = pd.to_numeric(matches['home_score'],  errors='coerce')
matches['away_score']  = pd.to_numeric(matches['away_score'],  errors='coerce')
matches['home_xg']     = pd.to_numeric(matches['home_xg'],     errors='coerce')
matches['away_xg']     = pd.to_numeric(matches['away_xg'],     errors='coerce')
matches['date']        = pd.to_datetime(matches['date'],        errors='coerce')

# ─── TEAM SUMMARY TABLE ──────────────────────────────────────
all_teams = set(matches['home_team'].unique()) | set(matches['away_team'].unique())
team_stats = []

for team in all_teams:
    home = matches[matches['home_team'] == team]
    away = matches[matches['away_team'] == team]
    played = len(home) + len(away)
    if played == 0:
        continue
    wins   = (home['result'] == 'home_win').sum() + (away['result'] == 'away_win').sum()
    draws  = (home['result'] == 'draw').sum()     + (away['result'] == 'draw').sum()
    losses = played - wins - draws
    gf     = pd.to_numeric(home['home_score'], errors='coerce').sum() + \
             pd.to_numeric(away['away_score'], errors='coerce').sum()
    ga     = pd.to_numeric(home['away_score'], errors='coerce').sum() + \
             pd.to_numeric(away['home_score'], errors='coerce').sum()
    tournaments_played = matches[
        (matches['home_team'] == team) | (matches['away_team'] == team)
    ]['year'].nunique()

    team_stats.append({
        'team':               team,
        'matches_played':     int(played),
        'wins':               int(wins),
        'draws':              int(draws),
        'losses':             int(losses),
        'goals_for':          float(gf),
        'goals_against':      float(ga),
        'goal_difference':    float(gf - ga),
        'win_pct':            round(wins / played * 100, 1),
        'tournaments_played': int(tournaments_played)
    })

team_summary = pd.DataFrame(team_stats).sort_values('win_pct', ascending=False)

# ─── YEARLY GOALS SUMMARY ────────────────────────────────────
yearly_summary = matches.groupby('year').agg(
    total_goals    = ('total_goals', 'sum'),
    avg_goals      = ('total_goals', 'mean'),
    total_matches  = ('match_id',    'count'),
    avg_attendance = ('attendance',  'mean')
).reset_index().merge(
    tournament[['year','host','champion','runner_up','top_scorer']],
    on='year', how='left'
)
yearly_summary['avg_goals']      = yearly_summary['avg_goals'].round(2)
yearly_summary['avg_attendance'] = yearly_summary['avg_attendance'].round(0)

# ─── MATCH RESULTS ENRICHED ──────────────────────────────────
matches_enriched = matches[[
    'match_id','year','date','round','venue',
    'home_team','away_team',
    'home_score','away_score','total_goals',
    'home_xg','away_xg','result','attendance'
]].copy()
matches_enriched['goal_diff']  = matches_enriched['home_score'] - matches_enriched['away_score']
matches_enriched['is_home_win'] = (matches_enriched['result'] == 'home_win').astype(int)
matches_enriched['is_draw']     = (matches_enriched['result'] == 'draw').astype(int)

# ─── RANKINGS LATEST ─────────────────────────────────────────
rankings_latest = rankings[rankings['snapshot_date'] == '2022-10-06'][[
    'team_name','team_code','association','rank','points'
]].sort_values('rank')

# ─── EXPORT TO EXCEL ─────────────────────────────────────────
output_path = 'dashboards/fifa_powerbi_data.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    matches_enriched.to_excel(writer, sheet_name='Matches',        index=False)
    team_summary.to_excel(    writer, sheet_name='Team_Summary',   index=False)
    yearly_summary.to_excel(  writer, sheet_name='Yearly_Summary', index=False)
    tournament.to_excel(      writer, sheet_name='Tournaments',    index=False)
    rankings_latest.to_excel( writer, sheet_name='Rankings',       index=False)
    schedule.to_excel(        writer, sheet_name='Schedule_2026',  index=False)

print(f"✅ Exported to {output_path}")
print(f"   Sheets: Matches({len(matches_enriched)}) | "
      f"Team_Summary({len(team_summary)}) | "
      f"Yearly_Summary({len(yearly_summary)}) | "
      f"Rankings({len(rankings_latest)}) | "
      f"Schedule({len(schedule)})")