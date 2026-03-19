import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json

# Set page config
st.set_page_config(
    page_title="Wikipedia Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper: Display Image Gallery
def display_report_gallery(folder_path, section_title):
    if os.path.exists(folder_path):
        st.subheader(section_title)
        images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            st.write("No report images found in this section.")
            return
            
        cols = st.columns(2)
        for idx, img_name in enumerate(images):
            with cols[idx % 2]:
                st.image(os.path.join(folder_path, img_name), 
                         caption=img_name.replace("_", " ").replace(".png", "").title(),
                         use_container_width=True)
    else:
        st.warning(f"Report folder not found: {folder_path}")

# Data Loading
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Sidebar Navigation
st.sidebar.title("Wikipedia Analysis")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/1200px-Wikipedia-logo-v2.svg.png", width=80)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", 
                        ["Pageview Analysis", "Editor Reports", "Top Articles", "Reddit Analysis", "Twitter Analysis"])

# --- Page 1: Pageview Analysis ---
if page == "Pageview Analysis":
    st.title("Pageview Analysis")
    
    # Interactive Trends
    df_yearly = load_csv("data/processed/en_wiki_pageviews_yearly.csv")
    if df_yearly is not None:
        st.subheader("Yearly Traffic Trends")
        fig = px.line(df_yearly, x='timestamp', y='views', markers=True, template="plotly_white")
        fig.update_traces(line_color='#333')
        st.plotly_chart(fig, use_container_width=True)
        
    df_ai = load_csv("data/processed/pageviews/ai_article_pageviews.csv")
    if df_ai is not None:
        st.subheader("AI Article Traffic Comparison")
        fig_ai = px.line(df_ai, x='timestamp', y='views', color='article', template="plotly_white")
        st.plotly_chart(fig_ai, use_container_width=True)

    # Gallery
    display_report_gallery("reports/pageviews_reports", "Detailed Traffic Reports")

# --- Page 2: Editor Reports ---
elif page == "Editor Reports":
    st.title("Editor Reports")
    st.markdown("Analysis of Wikipedia's community health and editor contribution patterns.")
    display_report_gallery("reports/editor_reports", "Contribution & Growth Patterns")

# --- Page 3: Top Articles ---
elif page == "Top Articles":
    st.title("Top Articles")
    
    df_top = load_csv("data/processed/top_articles/top_articles_2024_clean.csv")
    if df_top is not None:
        st.subheader("Trending Articles 2024")
        selected_article = st.selectbox("Select Article to Track", df_top['article'].unique()[:20])
        art_df = df_top[df_top['article'] == selected_article]
        fig_top = px.area(art_df, x='date', y='views', title=f"Views for {selected_article}", template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)

    display_report_gallery("reports/top_articles", "Attention & Spike Analysis")

# --- Page 4: Reddit Analysis ---
elif page == "Reddit Analysis":
    st.title("Reddit Analysis")
    
    # Detailed Interactive Charts
    st.subheader("Discussion Dynamics")
    df_reddit_sent = load_csv("data/processed/reddit_processed/reddit_with_topics.csv")
    if df_reddit_sent is not None:
        # Fallback for missing labels
        if 'vader_label' not in df_reddit_sent.columns and 'vader_compound' in df_reddit_sent.columns:
            df_reddit_sent['vader_label'] = df_reddit_sent['vader_compound'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
        
        target_label = 'roberta_label' if 'roberta_label' in df_reddit_sent.columns else 'vader_label'
        
        col1, col2 = st.columns(2)
        with col1:
            if target_label in df_reddit_sent.columns:
                fig_sent = px.histogram(df_reddit_sent, x=target_label, title=f"Sentiment Distribution ({target_label.split('_')[0].title()})", template="plotly_white")
                st.plotly_chart(fig_sent, use_container_width=True)
        with col2:
            if 'dominant_topic' in df_reddit_sent.columns:
                fig_topic = px.pie(df_reddit_sent, names='dominant_topic', title="Dominant Topics share", template="plotly_white")
                st.plotly_chart(fig_topic, use_container_width=True)

    # Gallery
    display_report_gallery("reports/reddit_reports", "Community Discussions & Sentiment Trends")

# --- Page 5: Twitter Analysis ---
elif page == "Twitter Analysis":
    st.title("Twitter Analysis")
    
    st.subheader("Social Engagement Patterns")
    df_twitter = load_csv("data/processed/twitter_processed/twitter_with_topics.csv")
    if df_twitter is not None:
        # Fallback for missing labels
        if 'vader_label' not in df_twitter.columns and 'vader_compound' in df_twitter.columns:
            df_twitter['vader_label'] = df_twitter['vader_compound'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
            
        col1, col2 = st.columns(2)
        with col1:
            if 'vader_label' in df_twitter.columns:
                fig_tw_sent = px.box(df_twitter, x='vader_label', y='vader_compound', title="Sentiment Consistency", template="plotly_white")
                st.plotly_chart(fig_tw_sent, use_container_width=True)
        with col2:
            if 'vader_compound' in df_twitter.columns and 'total_engagement' in df_twitter.columns:
                fig_tw_eng = px.scatter(df_twitter, x='vader_compound', y='total_engagement', hover_data=['text'] if 'text' in df_twitter.columns else None, 
                                       title="Sentiment vs Engagement", template="plotly_white")
                st.plotly_chart(fig_tw_eng, use_container_width=True)

    # Gallery
    display_report_gallery("reports/twitter_reports", "Engagement, Topics, and Wordclouds")
