import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text
from db.database import get_engine
from agents.agent import run_agent, get_agent_executor
from datetime import datetime


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Intelligence Agent",
    page_icon="📈",
    layout="wide"
)

# Cache agent - built once, reused across all reruns
@st.cache_resource
def load_agent():
    return get_agent_executor()

agent_executor = load_agent()


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_sentiment_by_category():
    engine = get_engine()
    query = text("""
        SELECT
            category,
            ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment,
            COUNT(*) AS total_articles,
            SUM(CASE WHEN sentiment_label='POSITIVE' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment_label='NEGATIVE' THEN 1 ELSE 0 END) AS negative
        FROM articles
        WHERE is_processed = TRUE
        GROUP BY category
        ORDER BY avg_sentiment DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=300)
def load_articles_over_time():
    engine = get_engine()
    query = text("""
        SELECT
            DATE(published_at AT TIME ZONE 'UTC') AS date,
            category,
            COUNT(*) AS article_count
        FROM articles
        GROUP BY DATE(published_at AT TIME ZONE 'UTC'), category
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["category"] = df["category"].str.title()
    return df


@st.cache_data(ttl=300)
def load_recent_articles(limit=15):
    engine = get_engine()
    query = text("""
        SELECT
            title, source_name, category,
            sentiment_label, sentiment_score, published_at, url
        FROM articles
        WHERE is_processed = TRUE
        ORDER BY published_at DESC
        LIMIT 15
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        return pd.DataFrame(result.fetchall(), columns=result.keys())


@st.cache_data(ttl=300)
def load_pipeline_logs():
    engine = get_engine()
    query = text("""
        SELECT run_at, articles_fetched, articles_inserted, status
        FROM pipeline_logs
        ORDER BY run_at DESC
        LIMIT 5
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        return pd.DataFrame(result.fetchall(), columns=result.keys())
    
@st.cache_data(ttl=300)
def load_top_entities(entity_type: str = "ORG", limit: int = 10):
    engine = get_engine()
    query = text("""
        SELECT entities FROM articles
        WHERE entities IS NOT NULL
        AND entities != '{}'
        AND entities != 'null'
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    import json
    from collections import Counter

    # Filter out noise — generic words, news wires, partial tokens
    blocklist = {
        "company", "companies", "data", "strong", "new", "inc", "ltd",
        "corp", "group", "fund", "market", "news", "prnewswire", "prn",
        "businesswire", "globenewswire", "reuters", "ap", "bfa law",
        "bfa", "the", "of", "ai", "ettech", "macdailynew"
    }

    counter = Counter()
    for row in rows:
        try:
            data = json.loads(row.entities)
            for name in data.get(entity_type, []):
                cleaned = name.strip()
                # Filter noisy tokens
                if len(cleaned) < 3:
                    continue
                if "##" in cleaned:
                    continue
                if cleaned.startswith("'"):
                    continue
                if cleaned.lower() in blocklist:
                    continue
                counter[cleaned.title()] += 1
        except Exception:
            continue

    top = counter.most_common(limit)
    return pd.DataFrame(top, columns=["entity", "count"])    


def sentiment_color(score):
    if score >= 0.2:
        return "🟢"
    elif score <= -0.2:
        return "🔴"
    else:
        return "🟡"


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stMetricLabel"] p {
        font-family: 'Georgia', serif !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #aaaaaa !important;
    }
    [data-testid="stDataFrame"] th {
        font-family: 'Georgia', serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stDataFrame"] td:nth-child(5) {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Market Intelligence")
    st.markdown("---")

    st.markdown("### Pipeline Status")
    logs = load_pipeline_logs()
    if not logs.empty:
        last = logs.iloc[0]
        st.success(f"Last run: {str(last['run_at'])[:16]}")
        st.metric("Articles fetched", last["articles_fetched"])
        st.metric("Articles inserted", last["articles_inserted"])
    else:
        st.warning("No pipeline runs logged yet.")

    st.markdown("---")
    st.markdown("### Auto-refresh")
    auto_refresh = st.toggle("Enable auto-refresh (5 min)", value=False)
    if auto_refresh:
        st.info("Dashboard will refresh every 5 minutes.")
        st.markdown(
            '<meta http-equiv="refresh" content="300">',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("🔄 Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()


# ── Main layout ────────────────────────────────────────────────────────────────
st.title("🧠 Market Intelligence Agent")
st.markdown("*Live news analytics powered by RAG + AI Agent*")
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
df_sent = load_sentiment_by_category()
df_sent["category"] = df_sent["category"].str.title()
total_articles = int(df_sent["total_articles"].sum())
most_negative = df_sent.loc[df_sent["avg_sentiment"].idxmin(), "category"]
most_positive = df_sent.loc[df_sent["avg_sentiment"].idxmax(), "category"]
overall_mood = df_sent["avg_sentiment"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📰 Total Articles", total_articles)
col2.metric("📊 Market Sentiment Score", f"{overall_mood:.3f}")
col3.metric("🔴 Most Negative Category", most_negative)
col4.metric("🟢 Most Positive Category", most_positive)

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment by Category")
    fig_bar = px.bar(
        df_sent,
        x="category",
        y="avg_sentiment",
        color="avg_sentiment",
        color_continuous_scale=[
            [0.0, "#ef4444"],
            [0.5, "#f59e0b"],
            [1.0, "#22c55e"]
        ],
        range_color=[-0.5, 0.5],
        labels={"avg_sentiment": "Avg Sentiment", "category": "Category"},
        text="avg_sentiment"
    )
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=350,
        margin=dict(t=20, b=20),
        xaxis=dict(
            title=dict(text="Category", font=dict(family="Georgia, serif", size=14, color="white")),
            tickfont=dict(family="Georgia, serif")
        ),
        yaxis=dict(
            title=dict(text="Avg Sentiment", font=dict(family="Georgia, serif", size=14, color="white")),
            tickfont=dict(family="Georgia, serif")
        )
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Positive vs Negative by Category")
    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(
        name="Negative",
        x=df_sent["category"],
        y=df_sent["negative"],
        marker_color="#ef4444"
    ))
    fig_stacked.add_trace(go.Bar(
        name="Positive",
        x=df_sent["category"],
        y=df_sent["positive"],
        marker_color="#22c55e"
    ))
    fig_stacked.update_layout(
        barmode="stack",
        height=350,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(
            title=dict(text="Category", font=dict(family="Georgia, serif", size=14, color="white")),
            tickfont=dict(family="Georgia, serif")
        ),
        yaxis=dict(
            title=dict(text="Article Count", font=dict(family="Georgia, serif", size=14, color="white")),
            tickfont=dict(family="Georgia, serif")
        )
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

# ── Articles over time ─────────────────────────────────────────────────────────
st.subheader("📅 Article Volume Over Time")
df_time = load_articles_over_time()
if not df_time.empty:
    fig_line = px.line(
        df_time,
        x="date",
        y="article_count",
        color="category",
        labels={"article_count": "Articles", "date": "Date"},
        height=300
    )
    fig_line.update_layout(
    margin=dict(t=20, b=20),
    height=300,
    xaxis=dict(
        tickformat="%b %d, %Y",
        dtick="D1",
        title=dict(text="Date", font=dict(family="Georgia, serif", size=14, color="white")),
        tickfont=dict(family="Georgia, serif")
    ),
    yaxis=dict(
        title=dict(text="Articles", font=dict(family="Georgia, serif", size=14, color="white")),
        tickfont=dict(family="Georgia, serif")
    )
)
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# ── Top Mentioned Organizations ───────────────────────────────────────────────
st.subheader("🏢 Most Mentioned Organizations")
df_orgs = load_top_entities("ORG", 10)
if not df_orgs.empty:
    fig_orgs = px.bar(
        df_orgs,
        x="count",
        y="entity",
        orientation="h",
        labels={"count": "Mentions", "entity": "Organization"},
        color="count",
        color_continuous_scale=[
            [0.0, "#3b82f6"],
            [1.0, "#8b5cf6"]
        ],
        text="count"
    )
    fig_orgs.update_traces(textposition="outside")
    fig_orgs.update_layout(
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=20, b=20),
        yaxis=dict(
            autorange="reversed",
            title=dict(text="Organization", font=dict(family="Georgia, serif", size=14, color="white")),
            tickfont=dict(family="Georgia, serif")
        ),
        xaxis=dict(
            title=dict(text="Mentions", font=dict(family="Georgia, serif", size=14, color="white")),
            tickfont=dict(family="Georgia, serif")
        )
    )
    st.plotly_chart(fig_orgs, use_container_width=True)

# ── Latest articles table ──────────────────────────────────────────────────────
st.subheader("🗞️ Latest Articles")
df_recent = load_recent_articles(15)
if not df_recent.empty:
    df_recent["mood"] = df_recent["sentiment_score"].apply(sentiment_color)
    df_recent["category"] = df_recent["category"].str.title()
    df_recent["published_at"] = pd.to_datetime(
        df_recent["published_at"]
    ).dt.strftime("%Y-%m-%d %H:%M")

    df_recent_display = df_recent[["mood", "title", "source_name", "category",
                                    "sentiment_score", "published_at"]].copy()
    df_recent_display.columns = df_recent_display.columns = ["SENTIMENT", "TITLE", "SOURCE", "CATEGORY", "SENTIMENT SCORE", "PUBLISHED AT"]
    df_recent_display["SENTIMENT SCORE"] = df_recent_display["SENTIMENT SCORE"].apply(lambda x: f"{x:.3f}")

    st.dataframe(
        df_recent_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "SENTIMENT": st.column_config.TextColumn("SENTIMENT", width="small"),
            "TITLE": st.column_config.TextColumn("TITLE", width="large"),
            "SENTIMENT SCORE": st.column_config.TextColumn("SENTIMENT SCORE", width="medium"),
        }
    )

st.markdown("---")

# ── Chat history display ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat interface ─────────────────────────────────────────────────────────────
st.subheader("💬 Ask the Market Intelligence Agent")
st.markdown("*Ask anything about the news, sentiment, trends, or specific topics.*")

if prompt := st.chat_input("e.g. What's the sentiment around AI companies?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(prompt, executor=agent_executor)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})