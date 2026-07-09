import pandas as pd
import numpy as np
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── CONNECTION ──────────────────────────────────────────────
connection_url = URL.create(
    drivername="postgresql",
    username="postgres",
    password="Infi@2k4",
    host="localhost",
    port=5432,
    database="fifa_analytics"
)
engine = create_engine(connection_url)

# ─── LOAD ────────────────────────────────────────────────────
print("Loading data...")
matches  = pd.read_sql("SELECT * FROM fact_matches", engine)
rankings = pd.read_sql("SELECT * FROM dim_rankings", engine)

# ─── BUILD HISTORICAL TEAM STATS ─────────────────────────────
print("Building historical team performance features...")

matches = matches.sort_values('date').reset_index(drop=True)

home_stats = matches[['date','home_team','home_score','away_score','result']].copy()
home_stats.columns = ['date','team','goals_for','goals_against','result']
home_stats['win']     = (home_stats['result'] == 'home_win').astype(int)
home_stats['draw']    = (home_stats['result'] == 'draw').astype(int)
home_stats['loss']    = (home_stats['result'] == 'away_win').astype(int)
home_stats['is_home'] = 1

away_stats = matches[['date','away_team','away_score','home_score','result']].copy()
away_stats.columns = ['date','team','goals_for','goals_against','result']
away_stats['win']     = (away_stats['result'] == 'away_win').astype(int)
away_stats['draw']    = (away_stats['result'] == 'draw').astype(int)
away_stats['loss']    = (away_stats['result'] == 'home_win').astype(int)
away_stats['is_home'] = 0

all_team_matches = pd.concat([home_stats, away_stats]).sort_values('date').reset_index(drop=True)

# Cumulative expanding stats per team (shift to prevent leakage)
def get_team_history(df):
    df = df.sort_values('date').copy()
    df['cum_matches']   = df.groupby('team').cumcount()
    df['cum_wins']      = df.groupby('team')['win'].cumsum()
    df['cum_goals_for'] = df.groupby('team')['goals_for'].cumsum()
    df['cum_goals_ag']  = df.groupby('team')['goals_against'].cumsum()
    for col in ['cum_matches','cum_wins','cum_goals_for','cum_goals_ag']:
        df[col] = df.groupby('team')[col].shift(1)
    df['hist_win_rate']  = df['cum_wins']      / df['cum_matches'].replace(0, np.nan)
    df['hist_avg_gf']    = df['cum_goals_for'] / df['cum_matches'].replace(0, np.nan)
    df['hist_avg_ga']    = df['cum_goals_ag']  / df['cum_matches'].replace(0, np.nan)
    df['hist_goal_diff'] = df['hist_avg_gf']   - df['hist_avg_ga']
    return df

all_team_matches = get_team_history(all_team_matches)

# ─── EXTRACT HOME & AWAY HIST FEATURES ───────────────────────
home_hist = all_team_matches[all_team_matches['is_home'] == 1][[
    'date','team','hist_win_rate','hist_avg_gf','hist_avg_ga','hist_goal_diff'
]].rename(columns={
    'team':           'home_team',
    'hist_win_rate':  'home_hist_win_rate',
    'hist_avg_gf':    'home_hist_avg_gf',
    'hist_avg_ga':    'home_hist_avg_ga',
    'hist_goal_diff': 'home_hist_goal_diff'
})

away_hist = all_team_matches[all_team_matches['is_home'] == 0][[
    'date','team','hist_win_rate','hist_avg_gf','hist_avg_ga','hist_goal_diff'
]].rename(columns={
    'team':           'away_team',
    'hist_win_rate':  'away_hist_win_rate',
    'hist_avg_gf':    'away_hist_avg_gf',
    'hist_avg_ga':    'away_hist_avg_ga',
    'hist_goal_diff': 'away_hist_goal_diff'
})

# ─── MERGE ALL FEATURES ──────────────────────────────────────
print("Merging features...")
df = matches.merge(home_hist, on=['date','home_team'], how='left')
df = df.merge(away_hist,     on=['date','away_team'],  how='left')

# Rankings
rankings_ref = rankings[rankings['snapshot_date'] == '2022-10-06'][
    ['team_name','rank','points']
]
df = df.merge(
    rankings_ref.rename(columns={
        'team_name':'home_team','rank':'home_rank','points':'home_points'
    }), on='home_team', how='left'
)
df = df.merge(
    rankings_ref.rename(columns={
        'team_name':'away_team','rank':'away_rank','points':'away_points'
    }), on='away_team', how='left'
)

# Round encoding
round_order = {
    'Group stage':1,'First group stage':1,'Second group stage':2,
    'First round':1,'Round of 16':2,'Quarter-finals':3,
    'Semi-finals':4,'Third-place match':5,'Final':6
}
df['round_num']    = df['round'].map(round_order).fillna(1)
df['is_host']      = (df['home_team'] == df['host']).astype(int)
df['rank_diff']    = df['home_rank']   - df['away_rank']
df['points_diff']  = df['home_points'] - df['away_points']

# Differential features
df['win_rate_diff']  = df['home_hist_win_rate']  - df['away_hist_win_rate']
df['goal_diff_diff'] = df['home_hist_goal_diff'] - df['away_hist_goal_diff']
df['avg_gf_diff']    = df['home_hist_avg_gf']    - df['away_hist_avg_gf']

# Target
df['home_win'] = (df['result'] == 'home_win').astype(int)

# ─── FEATURE SET ─────────────────────────────────────────────
features = [
    'rank_diff', 'points_diff',
    'is_host', 'round_num',
    'home_hist_win_rate',  'away_hist_win_rate',  'win_rate_diff',
    'home_hist_avg_gf',    'away_hist_avg_gf',    'avg_gf_diff',
    'home_hist_goal_diff', 'away_hist_goal_diff', 'goal_diff_diff'
]

# ─── TEMPORAL TRAIN/TEST SPLIT ───────────────────────────────
data = df[features + ['home_win','year']].dropna()

train = data[data['year'] <  2014]
test  = data[data['year'] >= 2014]

X_train, y_train = train[features], train['home_win']
X_test,  y_test  = test[features],  test['home_win']

# Scale
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {len(X_train)} matches (pre-2014)")
print(f"Test:  {len(X_test)}  matches (2014-2022)")

# ─── TRAIN & EVALUATE ────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=300, max_depth=6,
                                                   min_samples_leaf=10, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=300, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          eval_metric='logloss', verbosity=0,
                                          random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                       learning_rate=0.05, random_state=42)
}

results  = {}
print("\n── MODEL COMPARISON ──────────────────────────────────")

for name, model in models.items():
    Xtr = X_train_s if name == 'Logistic Regression' else X_train
    Xte = X_test_s  if name == 'Logistic Regression' else X_test

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)
    cv_acc = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy').mean()

    results[name] = {
        'model':       model,
        'accuracy':    acc,
        'auc':         auc,
        'cv_accuracy': cv_acc,
        'y_pred':      y_pred,
        'y_proba':     y_proba
    }
    print(f"\n{name}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  CV Accuracy: {cv_acc:.4f}")

# ─── BEST MODEL ──────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['auc'])
best      = results[best_name]
print(f"\n✅ Best model: {best_name} (AUC: {best['auc']:.4f})")
print(classification_report(y_test, best['y_pred'],
                             target_names=['No Home Win','Home Win']))

# ─── CONFUSION MATRIX + FEATURE IMPORTANCE ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['No Win','Win'], yticklabels=['No Win','Win'])
axes[0].set_title(f'Confusion Matrix — {best_name}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

if hasattr(best['model'], 'feature_importances_'):
    imp = pd.Series(best['model'].feature_importances_,
                    index=features).sort_values(ascending=True)
else:
    imp = pd.Series(np.abs(best['model'].coef_[0]),
                    index=features).sort_values(ascending=True)

axes[1].barh(imp.index, imp.values, color='#2d9e64')
axes[1].set_title(f'Feature Importance — {best_name}')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('data/processed/model_v2_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── SAVE MODEL & ARTIFACTS ──────────────────────────────────
joblib.dump(best['model'], 'models/best_model_v2.pkl')
joblib.dump(scaler,        'models/scaler_v2.pkl')
joblib.dump(features,      'models/features_v2.pkl')

# Save team median stats for Streamlit app
team_stats_for_app = []
for team in all_team_matches['team'].unique():
    rows = all_team_matches[
        all_team_matches['team'] == team
    ].dropna(subset=['hist_win_rate','hist_avg_gf','hist_goal_diff']).tail(10)
    if len(rows) > 0:
        team_stats_for_app.append({
            'team':      team,
            'win_rate':  rows['hist_win_rate'].median(),
            'avg_gf':    rows['hist_avg_gf'].median(),
            'goal_diff': rows['hist_goal_diff'].median()
        })

team_stats_df = pd.DataFrame(team_stats_for_app)
team_stats_df.to_csv('data/processed/team_stats_for_app.csv', index=False)
joblib.dump(team_stats_df, 'models/team_stats_v2.pkl')
print(f"\nTeam stats saved for {len(team_stats_df)} teams")
print("✅ All model artifacts saved to models/")

# ─── PREDICTION FUNCTION ─────────────────────────────────────
def predict_match(home_team, away_team, is_host=0, round_num=1):
    rankings_ref_local = rankings[rankings['snapshot_date'] == '2022-10-06'][
        ['team_name','rank','points']
    ]
    home_r = rankings_ref_local[rankings_ref_local['team_name'] == home_team]
    away_r = rankings_ref_local[rankings_ref_local['team_name'] == away_team]

    if home_r.empty or away_r.empty:
        print(f"  ⚠ Team not found in rankings — check spelling")
        return None

    h_rank, h_pts = home_r['rank'].values[0], home_r['points'].values[0]
    a_rank, a_pts = away_r['rank'].values[0], away_r['points'].values[0]

    # Use median of last 10 historical stat rows per team
    def get_team_stats(team):
        rows = all_team_matches[
            all_team_matches['team'] == team
        ].dropna(subset=['hist_win_rate','hist_avg_gf','hist_goal_diff']).tail(10)
        if len(rows) == 0:
            return None
        return {
            'win_rate':  rows['hist_win_rate'].median(),
            'avg_gf':    rows['hist_avg_gf'].median(),
            'goal_diff': rows['hist_goal_diff'].median()
        }

    home_s = get_team_stats(home_team)
    away_s = get_team_stats(away_team)

    if home_s is None or away_s is None:
        print(f"  ⚠ Insufficient history for {home_team} or {away_team}")
        return None

    feat = pd.DataFrame([{
        'rank_diff':            h_rank            - a_rank,
        'points_diff':          h_pts             - a_pts,
        'is_host':              is_host,
        'round_num':            round_num,
        'home_hist_win_rate':   home_s['win_rate'],
        'away_hist_win_rate':   away_s['win_rate'],
        'win_rate_diff':        home_s['win_rate']  - away_s['win_rate'],
        'home_hist_avg_gf':     home_s['avg_gf'],
        'away_hist_avg_gf':     away_s['avg_gf'],
        'avg_gf_diff':          home_s['avg_gf']    - away_s['avg_gf'],
        'home_hist_goal_diff':  home_s['goal_diff'],
        'away_hist_goal_diff':  away_s['goal_diff'],
        'goal_diff_diff':       home_s['goal_diff'] - away_s['goal_diff'],
    }])

    if best_name == 'Logistic Regression':
        feat_in = scaler.transform(feat)
    else:
        feat_in = feat

    proba  = best['model'].predict_proba(feat_in)[0]
    pred   = best['model'].predict(feat_in)[0]

    print(f"\n{'='*50}")
    print(f"  {home_team:20s} vs  {away_team}")
    print(f"{'='*50}")
    print(f"  Home win probability : {proba[1]*100:.1f}%")
    print(f"  Away win probability : {proba[0]*100:.1f}%")
    print(f"  Predicted outcome    : {'Home Win' if pred==1 else 'Away Win / Draw'}")
    print(f"  Rankings             : {home_team} #{h_rank} vs {away_team} #{a_rank}")
    print(f"  Home hist win rate   : {home_s['win_rate']:.2f} | "
          f"Away: {away_s['win_rate']:.2f}")
    print(f"{'='*50}")
    return proba

# ─── SAMPLE PREDICTIONS ──────────────────────────────────────
print("\n── SAMPLE PREDICTIONS ────────────────────────────────")
predict_match('Brazil',    'Argentina')
predict_match('France',    'England')
predict_match('Spain',     'Germany')
predict_match('Argentina', 'France',  round_num=6)
predict_match('Morocco',   'France',  round_num=4)