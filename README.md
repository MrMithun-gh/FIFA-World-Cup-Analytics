# FIFA World Cup Analytics

An end-to-end analytics platform for FIFA World Cup data (1930–2026): a SQL-backed ETL pipeline, machine learning match-outcome predictor, Power BI dashboard, and an interactive Streamlit app.

## Features

- **ETL Pipeline** — Cleans and loads raw match, ranking, and schedule CSVs into a star-schema PostgreSQL database (`dim_tournament`, `dim_team`, `dim_rankings`, `dim_schedule`, `fact_matches`).
- **SQL Analysis** — Prebuilt queries for win rates, goals per team, tournament summaries, host advantage, goals by round, biggest upsets, and champions.
- **ML Match Predictor** — Logistic Regression, Random Forest, and XGBoost models (two iterations) trained to predict match outcomes, with model comparison and feature-importance plots.
- **Streamlit App** (`app.py`) — Interactive "FIFA World Cup Intelligence Platform" with team performance analysis, tournament history, a match outcome predictor, and EDA visualizations (goal distribution, xG analysis, correlations).
- **Power BI Dashboard** — `dashboards/FIFA.pbix` built from exported, cleaned tables.

## Tech Stack

- **Data/ETL:** Python, pandas, SQLAlchemy, PostgreSQL
- **ML:** scikit-learn, XGBoost, joblib
- **App:** Streamlit, Plotly
- **BI:** Power BI

## Project Structure

```
FIFA-World-Cup-Analytics/
├── app.py                      # Streamlit dashboard app
├── src/
│   ├── etl_pipeline.py         # Raw CSV → PostgreSQL star schema
│   ├── sql_analysis.py         # Analytical SQL queries (Q1–Q7)
│   ├── ml_model.py             # First-pass outcome prediction models
│   ├── ml_model_v2.py          # Improved models w/ feature engineering
│   └── export_for_powerbi.py   # Export cleaned tables for Power BI
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_eda.ipynb
├── data/
│   ├── raw/                    # Source CSVs (matches, rankings, schedule)
│   └── processed/              # Query outputs, charts, model artifacts
└── dashboards/
    ├── FIFA.pbix
    └── fifa_powerbi_data.xlsx
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/MrMithun-gh/FIFA-World-Cup-Analytics.git
   cd FIFA-World-Cup-Analytics
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy sqlalchemy psycopg2-binary scikit-learn xgboost joblib matplotlib seaborn streamlit plotly
   ```

3. **Set up PostgreSQL** and update the DB connection settings (currently hardcoded in `src/etl_pipeline.py`, `src/ml_model*.py`, `src/sql_analysis.py`, `src/export_for_powerbi.py`, and `app.py`) — ideally move these to environment variables before running.

4. **Run the pipeline**
   ```bash
   python src/etl_pipeline.py       # load raw data into Postgres
   python src/sql_analysis.py       # generate analytical outputs
   python src/ml_model_v2.py        # train the predictor
   python src/export_for_powerbi.py # export tables for Power BI
   ```

5. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```

## Data Sources

- FIFA World Cup match results, 1930–2022
- FIFA rankings (2022 and 2026 snapshots)
- 2026 World Cup schedule

## License

No license specified yet.
