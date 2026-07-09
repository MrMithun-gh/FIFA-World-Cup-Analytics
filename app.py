import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a1628; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #0f3d24, #1a5c35);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2d9e64;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0c94a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a0c4b0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .predict-box {
        background: linear-gradient(135deg, #0f2744, #1a3a5c);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2d6ea0;
        text-align: center;
    }
    .win-prob {
        font-size: 3rem;
        font-weight: 700;
    }
    .stSelectbox label { color: #a0c4b0 !important; }
    h1, h2, h3 { color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)

# ─── DB CONNECTION ────────────────────────────────────────────
@st.cache_resource
def get_engine():
    connection_url = URL.create(
        drivername="postgresql",
        username="postgres",
        password="Infi@2k4",
        host="localhost",
        port=5432,
        database="fifa_analytics"
    )
    return create_engine(connection_url)

# ─── LOAD DATA ───────────────────────────────────────────────
@st.cache_data
def load_data():
    engine = get_engine()
    matches    = pd.read_sql("SELECT * FROM fact_matches",  engine)
    tournament = pd.read_sql("SELECT * FROM dim_tournament", engine)
    rankings   = pd.read_sql("SELECT * FROM dim_rankings",  engine)
    return matches, tournament, rankings

@st.cache_resource
def load_model():
    model      = joblib.load('models/best_model_v2.pkl')
    scaler     = joblib.load('models/scaler_v2.pkl')
    features   = joblib.load('models/features_v2.pkl')
    team_stats = joblib.load('models/team_stats_v2.pkl')
    return model, scaler, features, team_stats

matches, tournament, rankings = load_data()
model, scaler, features, team_stats = load_model()

# ─── PRECOMPUTE TEAM STATS ───────────────────────────────────
def compute_team_stats(matches):
    all_teams = set(matches['home_team'].unique()) | set(matches['away_team'].unique())
    stats = []
    for team in all_teams:
        home = matches[matches['home_team'] == team]
        away = matches[matches['away_team'] == team]
        played = len(home) + len(away)
        if played == 0:
            continue
        wins   = (home['result'] == 'home_win').sum() + (away['result'] == 'away_win').sum()
        draws  = (home['result'] == 'draw').sum()     + (away['result'] == 'draw').sum()
        losses = played - wins - draws
        gf     = home['home_score'].sum() + away['away_score'].sum()
        ga     = home['away_score'].sum() + away['home_score'].sum()
        stats.append({
            'team': team, 'played': int(played),
            'wins': int(wins), 'draws': int(draws), 'losses': int(losses),
            'goals_for': float(gf or 0), 'goals_against': float(ga or 0),
            'win_pct': round(wins / played * 100, 1),
            'goal_diff': float((gf or 0) - (ga or 0))
        })
    return pd.DataFrame(stats).sort_values('win_pct', ascending=False)

team_df = compute_team_stats(matches)

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FIFA Analytics")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Team Analysis",
        "🏆 Tournament History",
        "🤖 Match Predictor",
        "📈 EDA Insights"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** 1930–2022")
    st.markdown("**Matches:** 964")
    st.markdown("**Teams:** 86")
    st.markdown("**Model AUC:** 0.7303")

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("⚽ FIFA World Cup Intelligence Platform")
    st.markdown("*End-to-end analytics platform covering 1930–2022 World Cup data*")
    st.markdown("---")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_goals  = int(pd.to_numeric(matches['total_goals'], errors='coerce').sum())
    avg_goals    = round(pd.to_numeric(matches['total_goals'], errors='coerce').mean(), 2)
    home_win_pct = round((matches['result'] == 'home_win').mean() * 100, 1)
    top_team     = team_df.iloc[0]['team']
    tournaments  = tournament['year'].nunique()

    for col, val, label in zip(
        [col1, col2, col3, col4, col5],
        [964, total_goals, f"{avg_goals}", f"{home_win_pct}%", tournaments],
        ["Total Matches", "Total Goals", "Avg Goals/Match", "Home Win Rate", "Tournaments"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥇 Most World Cup Titles")
        champs = tournament.groupby('champion').size().reset_index(name='titles')
        champs = champs.sort_values('titles', ascending=True)
        fig = px.bar(champs, x='titles', y='champion', orientation='h',
                     color='titles', color_continuous_scale='Greens',
                     title="World Cup Winners")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', showlegend=False,
            coloraxis_showscale=False, height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Goals per Tournament Over Time")
        goals_yr = matches.copy()
        goals_yr['total_goals'] = pd.to_numeric(goals_yr['total_goals'], errors='coerce')
        goals_yr = goals_yr.groupby('year')['total_goals'].agg(['sum','mean']).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=goals_yr['year'], y=goals_yr['sum'],
                             name='Total Goals', marker_color='#2d9e64'), secondary_y=False)
        fig.add_trace(go.Scatter(x=goals_yr['year'], y=goals_yr['mean'].round(2),
                                 name='Avg per Match', mode='lines+markers',
                                 line=dict(color='#f0c94a', width=2)), secondary_y=True)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=400,
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig, use_container_width=True)

    # Result distribution
    st.subheader("⚖️ Overall Match Result Distribution")
    col1, col2 = st.columns(2)
    with col1:
        result_counts = matches['result'].value_counts()
        fig = px.pie(values=result_counts.values,
                     names=['Home Win','Away Win','Draw'],
                     color_discrete_sequence=['#2d9e64','#185fa5','#c8961a'],
                     hole=0.4)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏠 Host Nation Advantage")
        host_matches     = matches[matches['home_team'] == matches['host']]
        non_host_matches = matches[matches['home_team'] != matches['host']]
        host_win     = round((host_matches['result'] == 'home_win').mean() * 100, 1)
        non_host_win = round((non_host_matches['result'] == 'home_win').mean() * 100, 1)

        fig = go.Figure(go.Bar(
            x=['Host Nation', 'Non-Host'],
            y=[host_win, non_host_win],
            marker_color=['#f0c94a','#2d9e64'],
            text=[f"{host_win}%", f"{non_host_win}%"],
            textposition='outside'
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=350,
            yaxis=dict(range=[0,100], title='Win %')
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 2 — TEAM ANALYSIS
# ══════════════════════════════════════════════
elif page == "📊 Team Analysis":
    st.title("📊 Team Performance Analysis")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        min_matches = st.slider("Minimum matches played", 5, 30, 10)
    with col2:
        metric = st.selectbox("Rank by", ["win_pct", "goals_for", "goal_diff", "played"])

    filtered = team_df[team_df['played'] >= min_matches].head(20)
    filtered_sorted = filtered.sort_values(metric, ascending=True)

    fig = px.bar(filtered_sorted, x=metric, y='team', orientation='h',
                 color=metric, color_continuous_scale='Greens',
                 title=f"Top Teams by {metric.replace('_',' ').title()}")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=550,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Team Deep Dive")
    selected_team = st.selectbox("Select a team", sorted(team_df['team'].tolist()))

    t = team_df[team_df['team'] == selected_team].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [int(t['played']), int(t['wins']), f"{t['win_pct']}%", int(t['goal_diff'])],
        ["Matches", "Wins", "Win Rate", "Goal Diff"]
    ):
        col.metric(label, val)

    # Team performance over time
    team_by_year = []
    team_matches = matches[
        (matches['home_team'] == selected_team) |
        (matches['away_team'] == selected_team)
    ]
    for year in sorted(team_matches['year'].unique()):
        ym = team_matches[team_matches['year'] == year]
        hw = (ym[ym['home_team'] == selected_team]['result'] == 'home_win').sum()
        aw = (ym[ym['away_team'] == selected_team]['result'] == 'away_win').sum()
        played_yr = len(ym)
        team_by_year.append({
            'year': year,
            'wins': int(hw + aw),
            'played': played_yr,
            'win_pct': round((hw + aw) / played_yr * 100, 1) if played_yr > 0 else 0
        })

    yr_df = pd.DataFrame(team_by_year)
    if not yr_df.empty:
        fig = px.line(yr_df, x='year', y='win_pct', markers=True,
                      title=f"{selected_team} — Win % per Tournament",
                      color_discrete_sequence=['#f0c94a'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 3 — TOURNAMENT HISTORY
# ══════════════════════════════════════════════
elif page == "🏆 Tournament History":
    st.title("🏆 Tournament History")
    st.markdown("---")

    matches['total_goals'] = pd.to_numeric(matches['total_goals'], errors='coerce')
    matches['attendance']  = pd.to_numeric(matches['attendance'],  errors='coerce')

    tourn_stats = matches.groupby('year').agg(
        total_goals=('total_goals','sum'),
        avg_goals=('total_goals','mean'),
        avg_attendance=('attendance','mean'),
        total_matches=('match_id','count')
    ).reset_index().merge(tournament[['year','host','champion']], on='year')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Total Goals per Tournament",
            "Avg Goals per Match",
            "Avg Attendance per Tournament",
            "Matches per Tournament"
        ]
    )
    colors = ['#2d9e64','#f0c94a','#185fa5','#c8961a']
    for i, (col, row, c) in enumerate([
        ('total_goals',1,1), ('avg_goals',1,2),
        ('avg_attendance',2,1), ('total_matches',2,2)
    ]):
        fig.add_trace(
            go.Bar(x=tourn_stats['year'], y=tourn_stats[col],
                   marker_color=colors[i], showlegend=False),
            row=row, col=c
        )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Full Tournament Table")
    display_df = tourn_stats[['year','host','champion',
                               'total_matches','total_goals',
                               'avg_goals','avg_attendance']].copy()
    display_df['avg_goals']      = display_df['avg_goals'].round(2)
    display_df['avg_attendance'] = display_df['avg_attendance'].round(0).astype('Int64')
    display_df = display_df.sort_values('year', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# PAGE 4 — MATCH PREDICTOR
# ══════════════════════════════════════════════
elif page == "🤖 Match Predictor":
    st.title("🤖 Match Outcome Predictor")
    st.markdown("*Powered by Logistic Regression — AUC 0.7303*")
    st.markdown("---")

    available_teams = sorted(team_stats['team'].tolist())
    rankings_ref    = rankings[rankings['snapshot_date'] == '2022-10-06']

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox("🏠 Home Team", available_teams,
                                  index=available_teams.index('Brazil') if 'Brazil' in available_teams else 0)
    with col2:
        st.markdown("<br><br><h2 style='text-align:center;color:#f0c94a'>VS</h2>",
                    unsafe_allow_html=True)
    with col3:
        away_team = st.selectbox("✈️ Away Team", available_teams,
                                  index=available_teams.index('Argentina') if 'Argentina' in available_teams else 1)

    col1, col2 = st.columns(2)
    with col1:
        is_host  = st.checkbox("🏟️ Home team is host nation", value=False)
    with col2:
        round_map = {
            'Group Stage':1, 'Round of 16':2,
            'Quarter-Final':3, 'Semi-Final':4,
            'Third Place':5, 'Final':6
        }
        round_sel = st.selectbox("🏆 Match Stage", list(round_map.keys()))
        round_num = round_map[round_sel]

    if st.button("⚽ Predict Match", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams.")
        else:
            home_r = rankings_ref[rankings_ref['team_name'] == home_team]
            away_r = rankings_ref[rankings_ref['team_name'] == away_team]
            home_s = team_stats[team_stats['team'] == home_team]
            away_s = team_stats[team_stats['team'] == away_team]

            if home_r.empty or away_r.empty or home_s.empty or away_s.empty:
                st.error("Could not find ranking/stats data for one of the teams.")
            else:
                h_rank   = home_r['rank'].values[0]
                a_rank   = away_r['rank'].values[0]
                h_pts    = home_r['points'].values[0]
                a_pts    = away_r['points'].values[0]
                h_wr     = home_s['win_rate'].values[0]
                a_wr     = away_s['win_rate'].values[0]
                h_gf     = home_s['avg_gf'].values[0]
                a_gf     = away_s['avg_gf'].values[0]
                h_gd     = home_s['goal_diff'].values[0]
                a_gd     = away_s['goal_diff'].values[0]

                feat = pd.DataFrame([{
                    'rank_diff':            h_rank - a_rank,
                    'points_diff':          h_pts  - a_pts,
                    'is_host':              int(is_host),
                    'round_num':            round_num,
                    'home_hist_win_rate':   h_wr,
                    'away_hist_win_rate':   a_wr,
                    'win_rate_diff':        h_wr   - a_wr,
                    'home_hist_avg_gf':     h_gf,
                    'away_hist_avg_gf':     a_gf,
                    'avg_gf_diff':          h_gf   - a_gf,
                    'home_hist_goal_diff':  h_gd,
                    'away_hist_goal_diff':  a_gd,
                    'goal_diff_diff':       h_gd   - a_gd,
                }])

                feat_scaled = scaler.transform(feat)
                proba = model.predict_proba(feat_scaled)[0]

                st.markdown("---")
                c1, c2, c3 = st.columns(3)

                with c1:
                    color = "#2d9e64" if proba[1] > 0.5 else "#666"
                    st.markdown(f"""
                    <div class="predict-box">
                        <div style="font-size:1.1rem;color:#a0c4b0">{home_team}</div>
                        <div class="win-prob" style="color:{color}">{proba[1]*100:.1f}%</div>
                        <div style="color:#a0c4b0">Win Probability</div>
                        <div style="color:#f0c94a;margin-top:8px">Rank #{h_rank}</div>
                    </div>""", unsafe_allow_html=True)

                with c2:
                    predicted = "🏠 Home Win" if proba[1] > 0.5 else "✈️ Away Win / Draw"
                    conf = max(proba) * 100
                    st.markdown(f"""
                    <div class="predict-box" style="background:linear-gradient(135deg,#1a1a2e,#2d2d44)">
                        <div style="font-size:0.9rem;color:#a0a0c0;margin-bottom:8px">PREDICTION</div>
                        <div style="font-size:1.4rem;color:#f0c94a;font-weight:700">{predicted}</div>
                        <div style="color:#a0a0c0;margin-top:12px">Confidence</div>
                        <div style="font-size:1.8rem;color:#fff;font-weight:700">{conf:.1f}%</div>
                        <div style="color:#888;font-size:0.75rem;margin-top:8px">{round_sel}</div>
                    </div>""", unsafe_allow_html=True)

                with c3:
                    color = "#2d9e64" if proba[0] > 0.5 else "#666"
                    st.markdown(f"""
                    <div class="predict-box">
                        <div style="font-size:1.1rem;color:#a0c4b0">{away_team}</div>
                        <div class="win-prob" style="color:{color}">{proba[0]*100:.1f}%</div>
                        <div style="color:#a0c4b0">Win Probability</div>
                        <div style="color:#f0c94a;margin-top:8px">Rank #{a_rank}</div>
                    </div>""", unsafe_allow_html=True)

                # Probability bar
                st.markdown("<br>", unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[proba[1]*100, proba[0]*100],
                    y=[home_team, away_team],
                    orientation='h',
                    marker_color=['#2d9e64','#185fa5'],
                    text=[f"{proba[1]*100:.1f}%", f"{proba[0]*100:.1f}%"],
                    textposition='inside'
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=160,
                    xaxis=dict(range=[0,100], title='Win Probability (%)'),
                    margin=dict(l=10,r=10,t=10,b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats comparison table
                st.markdown("### 📊 Team Stats Comparison")
                comp_df = pd.DataFrame({
                    'Metric':       ['FIFA Rank','FIFA Points','Historical Win Rate',
                                     'Avg Goals Scored','Avg Goal Difference'],
                    home_team:      [h_rank, round(h_pts,1), f"{h_wr:.2f}",
                                     f"{h_gf:.2f}", f"{h_gd:.2f}"],
                    away_team:      [a_rank, round(a_pts,1), f"{a_wr:.2f}",
                                     f"{a_gf:.2f}", f"{a_gd:.2f}"]
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# PAGE 5 — EDA INSIGHTS
# ══════════════════════════════════════════════
elif page == "📈 EDA Insights":
    st.title("📈 EDA Insights")
    st.markdown("---")

    matches['total_goals'] = pd.to_numeric(matches['total_goals'], errors='coerce')
    matches['home_xg']     = pd.to_numeric(matches['home_xg'],     errors='coerce')
    matches['away_xg']     = pd.to_numeric(matches['away_xg'],     errors='coerce')

    tab1, tab2, tab3 = st.tabs(["Goal Distribution", "xG Analysis", "Correlation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(matches, x='total_goals', nbins=15,
                               title="Distribution of Goals per Match",
                               color_discrete_sequence=['#2d9e64'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            goals_by_round = matches.groupby('round')['total_goals'].mean().reset_index()
            goals_by_round = goals_by_round.sort_values('total_goals', ascending=True)
            fig = px.bar(goals_by_round, x='total_goals', y='round',
                         orientation='h', title="Avg Goals by Round",
                         color_discrete_sequence=['#c8961a'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        xg_data = matches.dropna(subset=['home_xg','away_xg'])
        if len(xg_data) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(xg_data, x='home_xg', y='home_score',
                                 color='result', title="Home xG vs Actual Goals",
                                 color_discrete_map={
                                     'home_win':'#2d9e64',
                                     'away_win':'#185fa5',
                                     'draw':'#c8961a'
                                 })
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                xg_data = xg_data.copy()
                xg_data['xg_diff'] = xg_data['home_xg'] - xg_data['away_xg']
                fig = px.box(xg_data, x='result', y='xg_diff',
                             title="xG Difference by Match Result",
                             color='result',
                             color_discrete_map={
                                 'home_win':'#2d9e64',
                                 'away_win':'#185fa5',
                                 'draw':'#c8961a'
                             })
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("xG data only available for recent tournaments (2018–2022)")

    with tab3:
        num_cols = ['home_score','away_score','total_goals','home_xg','away_xg']
        corr_data = matches[num_cols].copy()
        for c in num_cols:
            corr_data[c] = pd.to_numeric(corr_data[c], errors='coerce')
        corr_data['home_win'] = (matches['result'] == 'home_win').astype(int)
        corr_data['goal_diff'] = corr_data['home_score'] - corr_data['away_score']

        import plotly.figure_factory as ff
        corr_matrix = corr_data.corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True,
                        color_continuous_scale='RdYlGn',
                        title="Correlation Matrix")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=500
        )
        st.plotly_chart(fig, use_container_width=True)