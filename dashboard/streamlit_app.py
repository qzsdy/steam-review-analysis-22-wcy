from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(
    page_title="Steam Review Analysis Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.35rem 1.55rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(80, 132, 255, 0.18), rgba(124, 77, 255, 0.12));
        border: 1px solid rgba(150, 150, 150, 0.22);
        margin-bottom: 1.1rem;
    }
    .hero h1 { margin: 0; font-size: 2.15rem; font-weight: 800; letter-spacing: -0.02em; }
    .hero p { margin: 0.45rem 0 0 0; color: #8f8f8f; font-size: 1.02rem; line-height: 1.55; }
    .info-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: rgba(128,128,128,0.055);
        border: 1px solid rgba(128,128,128,0.18);
        margin: 0.45rem 0 1rem 0;
    }
    .insight-title { font-weight: 750; font-size: 1.02rem; margin-bottom: 0.35rem; }
    .muted { color: #8f8f8f; font-size: 0.92rem; line-height: 1.45; }
    div[data-testid="stMetric"] {
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.055);
        padding: 0.85rem 0.95rem;
    }
    .footer {
        margin-top: 1.3rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(128,128,128,0.18);
        color: #8f8f8f;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = [
    Path("./notebooks/outputs/dashboard_data/steam_games_cleaned_dashboard.csv"),
    Path("./outputs/dashboard_data/steam_games_cleaned_dashboard.csv"),
    BASE_DIR / ".." / "notebooks" / "outputs" / "dashboard_data" / "steam_games_cleaned_dashboard.csv",
    BASE_DIR / ".." / "outputs" / "dashboard_data" / "steam_games_cleaned_dashboard.csv",
    BASE_DIR / "outputs" / "dashboard_data" / "steam_games_cleaned_dashboard.csv",
    Path("./data/steam-store-games/steam.csv"),
]

@st.cache_data(show_spinner=False)
def load_data():
    for path in CANDIDATE_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            source = str(path)
            break
    else:
        st.error("No dashboard data file found. Please include steam_games_cleaned_dashboard.csv in the project repository.")
        st.stop()

    if "total_ratings" not in df.columns and {"positive_ratings", "negative_ratings"}.issubset(df.columns):
        df["total_ratings"] = df["positive_ratings"].fillna(0) + df["negative_ratings"].fillna(0)

    if "positive_rate" not in df.columns and {"positive_ratings", "total_ratings"}.issubset(df.columns):
        df["positive_rate"] = df["positive_ratings"] / df["total_ratings"].replace(0, pd.NA)

    if "release_year" not in df.columns and "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    if "primary_genre" not in df.columns and "genres" in df.columns:
        df["primary_genre"] = df["genres"].fillna("Unknown").astype(str).str.split(";").str[0]

    if "primary_tag" not in df.columns and "steamspy_tags" in df.columns:
        df["primary_tag"] = df["steamspy_tags"].fillna("Unknown").astype(str).str.split(";").str[0]

    if "price_bucket" not in df.columns and "price" in df.columns:
        def price_bucket(x):
            try:
                x = float(x)
            except Exception:
                x = 0.0
            if x <= 0: return "Free"
            if x < 5: return "0-5"
            if x < 10: return "5-10"
            if x < 20: return "10-20"
            if x < 40: return "20-40"
            return "40+"
        df["price_bucket"] = df["price"].apply(price_bucket)

    if "owners_mid" not in df.columns:
        df["owners_mid"] = pd.NA

    for col in ["price", "positive_rate", "total_ratings", "average_playtime", "owners_mid", "release_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, source

df, source = load_data()

def pct(x):
    return "N/A" if pd.isna(x) else f"{x:.2%}"

def compact(x):
    if pd.isna(x):
        return "N/A"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:,.0f}"

def style_fig(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=56, b=32),
        title=dict(x=0.02, xanchor="left", font=dict(size=19)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return fig


# ============================================================
# Review keyword word cloud helper
# ============================================================

REVIEW_TEXT_COLUMNS = [
    "review_text", "reviews", "user_reviews", "user_review",
    "review", "comment", "comments", "content", "text"
]

REVIEW_KEYWORD_FALLBACK_COLUMNS = [
    "steamspy_tags", "primary_tag", "genres", "primary_genre", "categories"
]

@st.cache_data(show_spinner=False)
def collect_review_wordcloud_text(dataframe, selected_columns):
    words = []

    for col in selected_columns:
        if col not in dataframe.columns:
            continue

        values = dataframe[col].dropna().astype(str)

        for value in values:
            # Steam fields usually use semicolons; this also supports commas and slashes.
            parts = re.split(r"[;,|/]+", value)
            for part in parts:
                token = re.sub(r"[^A-Za-z0-9+# ._-]", " ", part).strip()
                token = re.sub(r"\s+", " ", token)
                if not token:
                    continue
                if token.lower() in {"unknown", "nan", "none", "null"}:
                    continue
                words.append(token)

    return " ".join(words)


def draw_review_wordcloud(text, title):
    if not text.strip():
        st.info("No available text keywords under the current filters.")
        return

    wc = WordCloud(
        width=1400,
        height=650,
        background_color="white",
        colormap="Blues",
        max_words=160,
        collocations=False,
        prefer_horizontal=0.92,
        random_state=42,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=14)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.sidebar.markdown("## 🎛️ Filters")
team_name = st.sidebar.text_input("Team Name / 小组名称", "22_wcy")
st.sidebar.caption("Explore Steam games by genre, year, price, review count, and keyword.")

genres = sorted(df.get("primary_genre", pd.Series(["Unknown"])).dropna().astype(str).unique())
selected_genres = st.sidebar.multiselect("Primary genre", genres, default=genres[:12] if len(genres) > 12 else genres)

min_year = int(df["release_year"].dropna().min()) if "release_year" in df.columns and df["release_year"].notna().any() else 1990
max_year = int(df["release_year"].dropna().max()) if "release_year" in df.columns and df["release_year"].notna().any() else 2020
year_range = st.sidebar.slider("Release year range", min_year, max_year, (min_year, max_year))

price_min = float(df["price"].dropna().min()) if "price" in df.columns and df["price"].notna().any() else 0.0
price_max = float(df["price"].dropna().max()) if "price" in df.columns and df["price"].notna().any() else 100.0
price_range = st.sidebar.slider("Price range", price_min, price_max, (price_min, price_max))

min_reviews = st.sidebar.number_input(
    "Minimum review count",
    min_value=0,
    max_value=int(df["total_ratings"].fillna(0).max()) if "total_ratings" in df.columns else 100000,
    value=0,
    step=100,
)

keyword = st.sidebar.text_input("Search game name", "")

filtered = df.copy()
if "primary_genre" in filtered.columns and selected_genres:
    filtered = filtered[filtered["primary_genre"].astype(str).isin(selected_genres)]
if "release_year" in filtered.columns:
    filtered = filtered[(filtered["release_year"].fillna(min_year) >= year_range[0]) & (filtered["release_year"].fillna(max_year) <= year_range[1])]
if "price" in filtered.columns:
    filtered = filtered[(filtered["price"].fillna(0) >= price_range[0]) & (filtered["price"].fillna(0) <= price_range[1])]
if "total_ratings" in filtered.columns:
    filtered = filtered[filtered["total_ratings"].fillna(0) >= min_reviews]
if keyword.strip() and "name" in filtered.columns:
    filtered = filtered[filtered["name"].astype(str).str.contains(keyword.strip(), case=False, na=False)]

st.markdown(
    f"""
    <div class="hero">
        <h1>🎮 Steam Review Analysis Dashboard</h1>
        <p>
            Interactive exploration of Steam game metadata, pricing, genres, platforms, playtime,
            estimated owners, and user review sentiment.<br>
            <b>Team:</b> {team_name} &nbsp; | &nbsp; <b>Dataset:</b> Kaggle Steam Store Games
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Loaded data source: `{source}`")

if len(filtered) == 0:
    st.warning("No games match the current filters. Please relax the filters in the sidebar.")

total_games = len(filtered)
avg_price = filtered["price"].mean() if "price" in filtered.columns and len(filtered) else pd.NA
total_reviews = filtered["total_ratings"].sum() if "total_ratings" in filtered.columns and len(filtered) else pd.NA
avg_positive = filtered["positive_rate"].mean() if "positive_rate" in filtered.columns and len(filtered) else pd.NA
free_ratio = filtered["price"].fillna(0).eq(0).mean() if "price" in filtered.columns and len(filtered) else pd.NA

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Games", f"{total_games:,}")
m2.metric("Average Price", f"{avg_price:.2f}" if not pd.isna(avg_price) else "N/A")
m3.metric("Total Reviews", compact(total_reviews))
m4.metric("Positive Rate", pct(avg_positive))
m5.metric("Free Game Ratio", pct(free_ratio))

tab_overview, tab_genre, tab_price, tab_trend, tab_top, tab_wordcloud, tab_data = st.tabs(
    ["📌 Overview", "🎯 Genres & Tags", "💰 Price & Sentiment", "📈 Trends & Platforms", "🏆 Top Games", "☁️ Review Word Cloud", "📄 Data Explorer"]
)

with tab_overview:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Project overview")
        st.markdown(
            """
            <div class="info-card">
            This dashboard presents the major results of the Steam Review Analysis project.
            It summarizes cleaned Spark output and supports interactive exploration of market
            structure, pricing, user reviews, release trends, and popular games.
            </div>
            """,
            unsafe_allow_html=True,
        )
        overview = pd.DataFrame(
            [
                ("Storage", "Raw CSV files stored in HDFS; cleaned data saved as Parquet."),
                ("Processing", "PySpark DataFrame operations and Spark SQL queries."),
                ("Analysis", "Genre, tag, price, platform, playtime, and review sentiment analysis."),
                ("Deployment", "Streamlit dashboard deployed online for interactive presentation."),
            ],
            columns=["Area", "Implementation"],
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Filtered snapshot")
        snapshot = pd.DataFrame(
            {
                "Metric": ["Selected games", "Average price", "Total reviews", "Average positive rate", "Free-game ratio"],
                "Value": [f"{total_games:,}", f"{avg_price:.2f}" if not pd.isna(avg_price) else "N/A", compact(total_reviews), pct(avg_positive), pct(free_ratio)],
            }
        )
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    st.subheader("Key insights")
    i1, i2, i3 = st.columns(3)
    top_genre = filtered["primary_genre"].dropna().astype(str).value_counts().idxmax() if "primary_genre" in filtered.columns and len(filtered) else "N/A"
    most_reviewed = filtered.sort_values("total_ratings", ascending=False).iloc[0]["name"] if {"name", "total_ratings"}.issubset(filtered.columns) and len(filtered) else "N/A"
    best_bucket = filtered.groupby("price_bucket")["positive_rate"].mean().sort_values(ascending=False).index[0] if {"price_bucket", "positive_rate"}.issubset(filtered.columns) and len(filtered) else "N/A"

    for col, title, value, note in [
        (i1, "Dominant genre", top_genre, "Largest genre under the current filters."),
        (i2, "Most reviewed game", most_reviewed, "Based on total positive and negative ratings."),
        (i3, "Best price bucket", best_bucket, "Measured by average positive review ratio."),
    ]:
        with col:
            st.markdown(f"<div class='info-card'><div class='insight-title'>{title}</div><div>{value}</div><div class='muted'>{note}</div></div>", unsafe_allow_html=True)

with tab_genre:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Game count by primary genre")
        if "primary_genre" in filtered.columns and len(filtered):
            genre_count = filtered.groupby("primary_genre", as_index=False).size().rename(columns={"size": "game_count"}).sort_values("game_count", ascending=False).head(20)
            fig = px.bar(genre_count, x="game_count", y="primary_genre", orientation="h", title="Top Genres by Number of Games", text="game_count")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, height=540), use_container_width=True)

    with col2:
        st.subheader("Top SteamSpy tags")
        if "steamspy_tags" in filtered.columns and len(filtered):
            tags = filtered["steamspy_tags"].dropna().astype(str).str.split(";").explode().str.strip()
            tag_count = tags[tags != ""].value_counts().head(20).reset_index()
            tag_count.columns = ["tag", "count"]
            fig = px.bar(tag_count, x="count", y="tag", orientation="h", title="Top SteamSpy Tags", text="count")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, height=540), use_container_width=True)

    st.markdown(
        "<div class='info-card'><b>Interpretation.</b> Genre and tag distributions reveal the Steam catalogue structure. "
        "They show which game categories dominate the market and which user-facing labels attract the most attention.</div>",
        unsafe_allow_html=True,
    )

with tab_price:
    st.subheader("Price bucket and review sentiment")
    price_order = ["Free", "0-5", "5-10", "10-20", "20-40", "40+"]

    if {"price_bucket", "positive_rate"}.issubset(filtered.columns) and len(filtered):
        price_summary = filtered.groupby("price_bucket", as_index=False).agg(
            game_count=("name", "count") if "name" in filtered.columns else ("price_bucket", "count"),
            avg_positive_rate=("positive_rate", "mean"),
            avg_price=("price", "mean"),
            avg_reviews=("total_ratings", "mean"),
            avg_owners=("owners_mid", "mean"),
        )
        price_summary["price_bucket"] = pd.Categorical(price_summary["price_bucket"], categories=price_order, ordered=True)
        price_summary = price_summary.sort_values("price_bucket")

        col1, col2 = st.columns([1.05, 0.95], gap="large")
        with col1:
            fig = px.bar(
                price_summary,
                x="price_bucket",
                y="avg_positive_rate",
                title="Average Positive Rate by Price Bucket",
                text=price_summary["avg_positive_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else ""),
                hover_data=["game_count", "avg_price", "avg_reviews"],
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with col2:
            sample = filtered.dropna(subset=["price", "positive_rate", "total_ratings"])
            if len(sample):
                sample = sample.sample(min(3000, len(sample)), random_state=42)
                fig = px.scatter(
                    sample,
                    x="price",
                    y="positive_rate",
                    size="total_ratings",
                    color="primary_genre" if "primary_genre" in sample.columns else None,
                    hover_name="name" if "name" in sample.columns else None,
                    title="Price vs Positive Rate",
                    opacity=0.68,
                )
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(style_fig(fig), use_container_width=True)

        table = price_summary.copy()
        table["avg_positive_rate"] = table["avg_positive_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        table["avg_price"] = table["avg_price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        table["avg_reviews"] = table["avg_reviews"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
        table["avg_owners"] = table["avg_owners"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='info-card'><b>Interpretation.</b> This page compares price levels with review sentiment and review volume, "
            "highlighting how free, low-price, and mid-price games differ in user feedback.</div>",
            unsafe_allow_html=True,
        )

with tab_trend:
    st.subheader("Release trends and platform support")

    if "release_year" in filtered.columns and len(filtered):
        year_summary = filtered.dropna(subset=["release_year"]).groupby("release_year", as_index=False).agg(
            game_count=("release_year", "size"),
            avg_positive_rate=("positive_rate", "mean") if "positive_rate" in filtered.columns else ("release_year", "size"),
            free_ratio=("price", lambda s: (s.fillna(0) == 0).mean()) if "price" in filtered.columns else ("release_year", "size"),
        ).sort_values("release_year")

        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig = px.line(year_summary, x="release_year", y="game_count", markers=True, title="Games Released by Year")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with col2:
            fig = px.line(year_summary, x="release_year", y="free_ratio", markers=True, title="Free Game Ratio by Year")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        if "platforms" in filtered.columns and len(filtered):
            platforms = filtered["platforms"].dropna().astype(str).str.split(";").explode().str.strip()
            platform_count = platforms[platforms != ""].value_counts().reset_index()
            platform_count.columns = ["platform", "game_count"]
            fig = px.bar(platform_count, x="platform", y="game_count", title="Supported Platform Distribution", text="game_count")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with col4:
        if {"average_playtime", "positive_rate"}.issubset(filtered.columns) and len(filtered):
            bins = [-1, 0, 60, 300, 1000, float("inf")]
            labels = ["0", "1-59 min", "1-5 h", "5-16 h", "16 h+"]
            play = filtered.copy()
            play["playtime_bucket"] = pd.cut(play["average_playtime"].fillna(0), bins=bins, labels=labels)
            play_summary = play.groupby("playtime_bucket", observed=True)["positive_rate"].mean().reset_index()
            fig = px.bar(play_summary, x="playtime_bucket", y="positive_rate", title="Positive Rate by Playtime Bucket", text=play_summary["positive_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else ""))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown(
        "<div class='info-card'><b>Interpretation.</b> Release trends show catalogue growth over time, while platform and playtime charts reveal "
        "differences in technical support and player engagement.</div>",
        unsafe_allow_html=True,
    )

with tab_top:
    st.subheader("Top high-rating games")
    if {"positive_rate", "total_ratings"}.issubset(filtered.columns) and len(filtered):
        threshold = st.number_input("Review threshold for ranking", min_value=0, value=1000, step=500)
        top = filtered[filtered["total_ratings"].fillna(0) >= threshold].sort_values(["positive_rate", "total_ratings"], ascending=[False, False]).head(50).copy()
        columns = [c for c in ["name", "release_year", "price", "positive_rate", "total_ratings", "owners", "primary_genre", "developer", "publisher"] if c in top.columns]
        display = top[columns].copy()
        if "positive_rate" in display.columns:
            display["positive_rate"] = display["positive_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if "price" in display.columns:
            display["price"] = display["price"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        if "total_ratings" in display.columns:
            display["total_ratings"] = display["total_ratings"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        st.dataframe(display, use_container_width=True, hide_index=True)

        if len(top):
            chart_data = top.head(15)
            fig = px.bar(chart_data, x="positive_rate", y="name", orientation="h", title="Top 15 Games by Positive Rate", hover_data=["total_ratings", "price"])
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig, height=560), use_container_width=True)


with tab_wordcloud:
    st.subheader("User Review Keyword Word Cloud")

    actual_review_cols = [c for c in REVIEW_TEXT_COLUMNS if c in filtered.columns]
    fallback_cols = [c for c in REVIEW_KEYWORD_FALLBACK_COLUMNS if c in filtered.columns]

    if actual_review_cols:
        st.caption(
            "This word cloud is generated from available user review/comment text columns in the filtered data."
        )
        source_choice = st.multiselect(
            "Text columns used for word cloud",
            actual_review_cols,
            default=actual_review_cols,
        )
    else:
        st.caption(
            "The current Steam Store Games dataset does not include full free-text user comments. "
            "Therefore, this chart uses SteamSpy tags, genres and categories as review-related community keywords."
        )
        source_choice = st.multiselect(
            "Keyword columns used for word cloud",
            fallback_cols,
            default=fallback_cols,
        )

    review_text = collect_review_wordcloud_text(filtered, source_choice)
    draw_review_wordcloud(review_text, "User Review Keyword Word Cloud")

    st.markdown(
        "<div class='info-card'><b>Interpretation.</b> Larger words appear more frequently in the selected data. "
        "The word cloud provides an intuitive view of common community keywords and review-related themes under the current filters.</div>",
        unsafe_allow_html=True,
    )


with tab_data:
    st.subheader("Data explorer")
    st.markdown(
        "<div class='info-card'>Inspect the filtered dataset directly. The table is limited to 1,000 rows for responsiveness, "
        "but the download button exports all filtered rows.</div>",
        unsafe_allow_html=True,
    )
    default_columns = [c for c in ["appid", "name", "release_year", "price", "positive_rate", "total_ratings", "owners", "primary_genre", "developer", "publisher"] if c in filtered.columns]
    selected_columns = st.multiselect("Displayed columns", list(filtered.columns), default=default_columns)
    if selected_columns:
        preview = filtered[selected_columns].head(1000).copy()
        if "positive_rate" in preview.columns:
            preview["positive_rate"] = preview["positive_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download filtered data as CSV",
            data=filtered[selected_columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="steam_filtered_dashboard_data.csv",
            mime="text/csv",
        )

st.markdown(f"<div class='footer'>Steam Review Analysis Dashboard · Team {team_name} · Built with Streamlit, Pandas and Plotly</div>", unsafe_allow_html=True)
