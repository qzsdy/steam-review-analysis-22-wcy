import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Steam Review Analysis Dashboard", page_icon="🎮", layout="wide")
st.title("🎮 Steam Review Analysis Dashboard")

team_name = st.sidebar.text_input("Team Name / 小组名称", "22_wcy")
st.sidebar.markdown(f"**Current Team:** {team_name}")
st.sidebar.markdown("This dashboard is the optional bonus part of the Big Data Final Project.")

BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = [
    # GitHub / Streamlit Cloud 部署时，从项目根目录找
    Path("./notebooks/outputs/dashboard_data/steam_games_cleaned_dashboard.csv"),

    # 本地运行 dashboard/streamlit_app.py 时，从 dashboard 目录往上找 notebooks
    BASE_DIR / ".." / "notebooks" / "outputs" / "dashboard_data" / "steam_games_cleaned_dashboard.csv",

    # 兼容原来的 outputs 目录
    BASE_DIR / ".." / "outputs" / "dashboard_data" / "steam_games_cleaned_dashboard.csv",
    BASE_DIR / "outputs" / "dashboard_data" / "steam_games_cleaned_dashboard.csv",
    Path("./outputs/dashboard_data/steam_games_cleaned_dashboard.csv"),

    # 最后兜底：原始 steam.csv
    Path("./data/steam-store-games/steam.csv"),
]

@st.cache_data
def load_data():
    for path in CANDIDATE_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            source = str(path)
            break
    else:
        st.error("No data file found. Please run the notebook first, or place steam.csv under ./data/steam-store-games/")
        st.stop()

    if "positive_rate" not in df.columns and {"positive_ratings", "negative_ratings"}.issubset(df.columns):
        df["total_ratings"] = df["positive_ratings"].fillna(0) + df["negative_ratings"].fillna(0)
        df["positive_rate"] = df["positive_ratings"] / df["total_ratings"].replace(0, pd.NA)

    if "price_bucket" not in df.columns and "price" in df.columns:
        def price_bucket(x):
            try:
                x = float(x)
            except Exception:
                x = 0
            if x <= 0:
                return "Free"
            if x < 5:
                return "0-5"
            if x < 10:
                return "5-10"
            if x < 20:
                return "10-20"
            if x < 40:
                return "20-40"
            return "40+"
        df["price_bucket"] = df["price"].apply(price_bucket)

    if "release_year" not in df.columns and "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    if "primary_genre" not in df.columns and "genres" in df.columns:
        df["primary_genre"] = df["genres"].fillna("Unknown").astype(str).str.split(";").str[0]

    if "total_ratings" not in df.columns:
        df["total_ratings"] = 0

    return df, source

df, source = load_data()
st.caption(f"Data source: {source}")

genres = sorted([g for g in df.get("primary_genre", pd.Series(["Unknown"])).dropna().unique()])
selected_genres = st.sidebar.multiselect("Select genres", genres, default=genres[:10] if len(genres) > 10 else genres)

min_year = int(df["release_year"].dropna().min()) if "release_year" in df.columns and df["release_year"].notna().any() else 1990
max_year = int(df["release_year"].dropna().max()) if "release_year" in df.columns and df["release_year"].notna().any() else 2020
year_range = st.sidebar.slider("Release year range", min_year, max_year, (min_year, max_year))

filtered = df.copy()
if "primary_genre" in filtered.columns and selected_genres:
    filtered = filtered[filtered["primary_genre"].isin(selected_genres)]
if "release_year" in filtered.columns:
    filtered = filtered[(filtered["release_year"].fillna(min_year) >= year_range[0]) & (filtered["release_year"].fillna(max_year) <= year_range[1])]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", f"{len(filtered):,}")
c2.metric("Avg Price", f"{filtered['price'].mean():.2f}" if "price" in filtered.columns else "N/A")
c3.metric("Total Reviews", f"{filtered['total_ratings'].sum():,.0f}" if "total_ratings" in filtered.columns else "N/A")
c4.metric("Avg Positive Rate", f"{filtered['positive_rate'].mean():.2%}" if "positive_rate" in filtered.columns else "N/A")

tab1, tab2, tab3, tab4 = st.tabs(["Genre & Tags", "Price & Rating", "Year Trend", "Top Games"])

with tab1:
    st.subheader("Game Count by Primary Genre")
    if "primary_genre" in filtered.columns:
        genre_count = filtered.groupby("primary_genre", as_index=False).size().sort_values("size", ascending=False).head(20)
        fig = px.bar(genre_count, x="size", y="primary_genre", orientation="h", title="Top Genres by Game Count")
        st.plotly_chart(fig, use_container_width=True)

    if "steamspy_tags" in filtered.columns:
        st.subheader("Top SteamSpy Tags")
        tags = filtered["steamspy_tags"].dropna().astype(str).str.split(";").explode().str.strip()
        tag_count = tags[tags != ""].value_counts().head(20).reset_index()
        tag_count.columns = ["tag", "count"]
        fig = px.bar(tag_count, x="count", y="tag", orientation="h", title="Top SteamSpy Tags")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Price Bucket vs Positive Rate")
    if {"price_bucket", "positive_rate"}.issubset(filtered.columns):
        price_order = ["Free", "0-5", "5-10", "10-20", "20-40", "40+"]
        price_summary = filtered.groupby("price_bucket", as_index=False).agg(
            game_count=("name", "count") if "name" in filtered.columns else ("appid", "count"),
            avg_positive_rate=("positive_rate", "mean"),
            avg_price=("price", "mean") if "price" in filtered.columns else ("positive_rate", "mean"),
            avg_reviews=("total_ratings", "mean") if "total_ratings" in filtered.columns else ("positive_rate", "mean"),
        )
        price_summary["price_bucket"] = pd.Categorical(price_summary["price_bucket"], categories=price_order, ordered=True)
        price_summary = price_summary.sort_values("price_bucket")
        fig = px.bar(price_summary, x="price_bucket", y="avg_positive_rate", hover_data=["game_count", "avg_price", "avg_reviews"])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(price_summary, use_container_width=True)

with tab3:
    st.subheader("Release Year Trend")
    if "release_year" in filtered.columns:
        year_summary = filtered.dropna(subset=["release_year"]).groupby("release_year", as_index=False).agg(
            game_count=("release_year", "size"),
            avg_price=("price", "mean") if "price" in filtered.columns else ("release_year", "size"),
            avg_positive_rate=("positive_rate", "mean") if "positive_rate" in filtered.columns else ("release_year", "size"),
        )
        st.plotly_chart(px.line(year_summary, x="release_year", y="game_count", markers=True, title="Games Released by Year"), use_container_width=True)
        st.plotly_chart(px.line(year_summary, x="release_year", y="avg_positive_rate", markers=True, title="Average Positive Rate by Year"), use_container_width=True)

with tab4:
    st.subheader("Top High-Rating Games")
    cols = [c for c in ["name", "release_year", "price", "positive_rate", "total_ratings", "owners", "primary_genre", "developer", "publisher"] if c in filtered.columns]
    if {"positive_rate", "total_ratings"}.issubset(filtered.columns):
        top = filtered[filtered["total_ratings"].fillna(0) >= 1000].sort_values(["positive_rate", "total_ratings"], ascending=[False, False]).head(50)
        st.dataframe(top[cols], use_container_width=True)
    else:
        st.dataframe(filtered.head(50), use_container_width=True)

st.markdown("---")
st.markdown(f"**Team Name:** {team_name}")
st.markdown("Dashboard generated for the Big Data Final Evaluation Project.")