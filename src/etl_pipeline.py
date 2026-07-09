import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# ─── DB CONNECTION ───────────────────────────────────────────
DB_USER     = "postgres"
DB_PASSWORD = "Infi%402k4"   # ← change this
DB_HOST     = "localhost"
DB_PORT     = "5432"
DB_NAME     = "fifa_analytics"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ─── LOAD RAW FILES ──────────────────────────────────────────
print("Loading raw CSV files...")
matches     = pd.read_csv('data/raw/matches_1930_2022.csv')
world_cup   = pd.read_csv('data/raw/world_cup.csv')
ranking_22  = pd.read_csv('data/raw/fifa_ranking_2022-10-06.csv')
ranking_26  = pd.read_csv('data/raw/fifa_ranking_2026-06-08.csv')
schedule    = pd.read_csv('data/raw/schedule_2026.csv')

# ─── CLEAN: DIM_TOURNAMENT ───────────────────────────────────
print("Building dim_tournament...")
dim_tournament = world_cup.rename(columns={
    'Year':         'year',
    'Host':         'host',
    'Teams':        'teams',
    'Champion':     'champion',
    'Runner-Up':    'runner_up',
    'TopScorrer':   'top_scorer',
    'Attendance':   'total_attendance',
    'AttendanceAvg':'avg_attendance',
    'Matches':      'total_matches'
})

# ─── CLEAN: DIM_TEAM ─────────────────────────────────────────
print("Building dim_team...")
home_teams = matches[['home_team']].rename(columns={'home_team': 'team_name'})
away_teams = matches[['away_team']].rename(columns={'away_team': 'team_name'})
all_teams  = pd.concat([home_teams, away_teams]).drop_duplicates().reset_index(drop=True)

# Merge with ranking to get team_code and association
ranking_ref = ranking_22[['team', 'team_code', 'association']].rename(
    columns={'team': 'team_name'}
)
dim_team = all_teams.merge(ranking_ref, on='team_name', how='left')
dim_team['team_id'] = range(1, len(dim_team) + 1)
dim_team = dim_team[['team_id', 'team_name', 'team_code', 'association']]

# ─── CLEAN: FACT_MATCHES ─────────────────────────────────────
print("Building fact_matches...")
fact_matches = matches[[
    'Year', 'Date', 'Round', 'Venue', 'Host',
    'home_team', 'away_team',
    'home_score', 'away_score',
    'home_xg', 'away_xg',
    'home_penalty', 'away_penalty',
    'home_red_card', 'away_red_card',
    'home_yellow_red_card', 'away_yellow_red_card',
    'Attendance', 'Referee'
]].copy()

fact_matches.columns = [
    'year', 'date', 'round', 'venue', 'host',
    'home_team', 'away_team',
    'home_score', 'away_score',
    'home_xg', 'away_xg',
    'home_penalty', 'away_penalty',
    'home_red_cards', 'away_red_cards',
    'home_yellow_red_cards', 'away_yellow_red_cards',
    'attendance', 'referee'
]

# Derive result column
fact_matches['result'] = np.where(
    fact_matches['home_score'] > fact_matches['away_score'], 'home_win',
    np.where(fact_matches['home_score'] < fact_matches['away_score'], 'away_win', 'draw')
)

# Total goals
fact_matches['total_goals'] = fact_matches['home_score'] + fact_matches['away_score']

# Clean attendance
fact_matches['attendance'] = pd.to_numeric(
    fact_matches['attendance'].astype(str).str.replace(',', ''), errors='coerce'
)

fact_matches['date'] = pd.to_datetime(fact_matches['date'], errors='coerce')
fact_matches['match_id'] = range(1, len(fact_matches) + 1)

# ─── CLEAN: DIM_RANKINGS ─────────────────────────────────────
print("Building dim_rankings...")
ranking_22['snapshot_date'] = '2022-10-06'
ranking_26_clean = ranking_26.drop(columns=['rated_matches'])
ranking_26_clean['snapshot_date'] = '2026-06-08'

dim_rankings = pd.concat([ranking_22, ranking_26_clean], ignore_index=True)
dim_rankings.columns = [
    'team_name', 'team_code', 'association',
    'rank', 'previous_rank', 'points', 'previous_points', 'snapshot_date'
]

# ─── CLEAN: DIM_SCHEDULE ─────────────────────────────────────
print("Building dim_schedule...")
dim_schedule = schedule[[
    'Year', 'Round', 'Date', 'Time', 'home_team', 'away_team'
]].rename(columns={
    'Year':      'year',
    'Round':     'round',
    'Date':      'date',
    'Time':      'kickoff_time',
    'home_team': 'home_team',
    'away_team': 'away_team'
})

# ─── LOAD TO POSTGRESQL ──────────────────────────────────────
print("\nLoading to PostgreSQL...")

tables = {
    'dim_tournament': dim_tournament,
    'dim_team':       dim_team,
    'dim_rankings':   dim_rankings,
    'dim_schedule':   dim_schedule,
    'fact_matches':   fact_matches,
}

for table_name, df in tables.items():
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"  ✓ {table_name}: {len(df)} rows loaded")

print("\n✅ ETL complete! All tables loaded into fifa_analytics database.")