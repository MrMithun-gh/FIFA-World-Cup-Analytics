import pandas as pd
import numpy as np
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
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

# ─── LOAD DATA ───────────────────────────────────────────────
print("Loading data...")
matches  = pd.read_sql("SELECT * FROM fact_matches",  engine)
rankings = pd.read_sql("SELECT * FROM dim_rankings",  engine)

# ─── FEATURE ENGINEERING ─────────────────────────────────────
print("Engineering features...")

# Use 2022 rankings as reference for all historical matches
rankings_ref = rankings[rankings['snapshot_date'] == '2022-10-06'][
    ['team_name', 'rank', 'points']
].copy()

# Merge home team rankings
df = matches.merge(
    rankings_ref.rename(columns={'team_name': 'home_team',
                                  'rank':      'home_rank',
                                  'points':    'home_points'}),
    on='home_team', how='left'
)

# Merge away team rankings
df = df.merge(
    rankings_ref.rename(columns={'team_name': 'away_team',
                                  'rank':      'away_rank',
                                  'points':    'away_points'}),
    on='away_team', how='left'
)

# Derived features
df['rank_diff']        = df['home_rank']   - df['away_rank']
df['points_diff']      = df['home_points'] - df['away_points']
df['is_host']          = (df['home_team']  == df['host']).astype(int)
df['home_xg']          = pd.to_numeric(df['home_xg'], errors='coerce')
df['away_xg']          = pd.to_numeric(df['away_xg'], errors='coerce')
df['xg_diff']          = df['home_xg'] - df['away_xg']
df['attendance_clean'] = pd.to_numeric(df['attendance'], errors='coerce')

# Encode round
round_order = {
    'Group stage': 1, 'First group stage': 1, 'Second group stage': 2,
    'First round': 1, 'Round of 16': 2, 'Quarter-finals': 3,
    'Semi-finals': 4, 'Third-place match': 5, 'Final': 6
}
df['round_num'] = df['round'].map(round_order).fillna(1)

# Target variable — binary: did home team win?
df['home_win'] = (df['result'] == 'home_win').astype(int)

# ─── FEATURE SET ─────────────────────────────────────────────
# Set A — Pre-match features only (usable for real predictions)
pre_match_features = [
    'rank_diff', 'points_diff', 'is_host', 'round_num'
]

# Set B — Include xg (post-match, better for analysis)
full_features = [
    'rank_diff', 'points_diff', 'is_host',
    'round_num', 'xg_diff'
]

# ─── PREPARE DATASETS ────────────────────────────────────────
def prepare_data(feature_cols):
    data = df[feature_cols + ['home_win']].dropna()
    X = data[feature_cols]
    y = data['home_win']
    return train_test_split(X, y, test_size=0.2, random_state=42)

print("\n── PRE-MATCH FEATURES (rank + host + round) ──")
X_train, X_test, y_train, y_test = prepare_data(pre_match_features)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─── TRAIN MODELS ────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=200, random_state=42,
                                         eval_metric='logloss', verbosity=0)
}

results = {}
print("\n── MODEL COMPARISON ──────────────────────────────────")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_acc  = cross_val_score(model, X_train, y_train,
                               cv=5, scoring='accuracy').mean()

    results[name] = {
        'model': model, 'accuracy': acc,
        'auc': auc, 'cv_accuracy': cv_acc,
        'y_pred': y_pred, 'y_proba': y_proba
    }
    print(f"\n{name}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  CV Accuracy: {cv_acc:.4f}")

# ─── BEST MODEL ──────────────────────────────────────────────
best_name  = max(results, key=lambda x: results[x]['auc'])
best       = results[best_name]
print(f"\n✅ Best model: {best_name} (AUC: {best['auc']:.4f})")

print(f"\nClassification Report — {best_name}:")
print(classification_report(y_test, best['y_pred'],
                             target_names=['No Home Win', 'Home Win']))

# ─── FEATURE IMPORTANCE ──────────────────────────────────────
print("\n── FEATURE IMPORTANCE ────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Random Forest importance
rf_model = results['Random Forest']['model']
rf_imp   = pd.Series(rf_model.feature_importances_,
                      index=pre_match_features).sort_values(ascending=True)
axes[0].barh(rf_imp.index, rf_imp.values, color='#185fa5')
axes[0].set_title('Random Forest — Feature Importance')
axes[0].set_xlabel('Importance')

# XGBoost importance
xgb_model = results['XGBoost']['model']
xgb_imp   = pd.Series(xgb_model.feature_importances_,
                       index=pre_match_features).sort_values(ascending=True)
axes[1].barh(xgb_imp.index, xgb_imp.values, color='#2d9e64')
axes[1].set_title('XGBoost — Feature Importance')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('data/processed/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── MODEL COMPARISON CHART ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(results.keys())
accuracies  = [results[m]['accuracy']    for m in model_names]
aucs        = [results[m]['auc']         for m in model_names]
cv_accs     = [results[m]['cv_accuracy'] for m in model_names]

x = np.arange(len(model_names))
w = 0.25
ax.bar(x - w,   accuracies, w, label='Test Accuracy', color='#185fa5')
ax.bar(x,       aucs,       w, label='AUC-ROC',       color='#2d9e64')
ax.bar(x + w,   cv_accs,    w, label='CV Accuracy',   color='#c8961a')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Accuracy vs AUC-ROC vs CV Accuracy')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('data/processed/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── SAVE BEST MODEL ─────────────────────────────────────────
joblib.dump(best['model'], f'models/best_model_{best_name.replace(" ", "_")}.pkl')
joblib.dump(pre_match_features, 'models/feature_names.pkl')
print(f"\n✅ Model saved to models/")

# ─── PREDICTION FUNCTION ─────────────────────────────────────
def predict_match(home_team, away_team, is_host=0, round_num=1):
    home_r = rankings_ref[rankings_ref['team_name'] == home_team]
    away_r = rankings_ref[rankings_ref['team_name'] == away_team]

    if home_r.empty or away_r.empty:
        print(f"Team not found in rankings. Check spelling.")
        return

    home_rank   = home_r['rank'].values[0]
    away_rank   = away_r['rank'].values[0]
    home_points = home_r['points'].values[0]
    away_points = away_r['points'].values[0]

    features = pd.DataFrame([{
        'rank_diff':   home_rank   - away_rank,
        'points_diff': home_points - away_points,
        'is_host':     is_host,
        'round_num':   round_num
    }])

    model   = best['model']
    proba   = model.predict_proba(features)[0]
    outcome = model.predict(features)[0]

    print(f"\n{'='*45}")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*45}")
    print(f"  Home win probability:  {proba[1]*100:.1f}%")
    print(f"  Away win probability:  {proba[0]*100:.1f}%")
    print(f"  Predicted outcome:     {'Home Win' if outcome == 1 else 'Away Win / Draw'}")
    print(f"  Home rank: {home_rank} | Away rank: {away_rank}")
    print(f"{'='*45}")

# ─── TEST PREDICTIONS ────────────────────────────────────────
print("\n── SAMPLE PREDICTIONS ────────────────────────────────")
predict_match('Brazil',    'Argentina')
predict_match('France',    'England')
predict_match('Spain',     'Germany')
predict_match('Argentina', 'France',  is_host=0, round_num=6)