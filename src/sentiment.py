import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader_sentiment(df, text_col="combined_text"):
    """run vader on combined_text (title + body). adds compound score, component scores, and label."""
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()

    # fall back to building combined text if not already present
    if text_col not in df.columns or df[text_col].isnull().all():
        if "title" in df.columns:
            df["combined_text"] = df["title"].fillna("") + " " + df.get(text_col, pd.Series([""] * len(df))).fillna("")
        else:
            df["combined_text"] = df[text_col].fillna("")
        text_col = "combined_text"

    scores = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x)))
    df["vader_compound"] = scores.apply(lambda x: x["compound"])
    df["vader_pos"] = scores.apply(lambda x: x["pos"])
    df["vader_neg"] = scores.apply(lambda x: x["neg"])
    df["vader_neu"] = scores.apply(lambda x: x["neu"])
    df["vader_label"] = df["vader_compound"].apply(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )
    return df


def roberta_sentiment(df, text_col="combined_text", batch_size=32):
    """run cardiffnlp twitter-roberta sentiment on text. truncates to 500 chars before tokenizing."""
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length=512,
        truncation=True
    )

    df = df.copy()

    if text_col not in df.columns:
        text_col = "combined_text"

    texts = df[text_col].fillna("").tolist()
    texts = [t[:500] for t in texts]

    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = sentiment_pipeline(batch)
        all_results.extend(results)

    df["roberta_label"] = [r["label"].lower() for r in all_results]
    df["roberta_score"] = [r["score"] for r in all_results]

    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["roberta_numeric"] = df["roberta_label"].map(label_map).fillna(0).astype(int)

    return df


def aggregate_sentiment_by_month(df, date_col="date", min_posts=5):
    """aggregate sentiment scores to monthly level.
    months with fewer than min_posts posts are flagged as unreliable."""
    df = df.copy()
    df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M")

    agg = df.groupby("month").agg(
        vader_mean=("vader_compound", "mean"),
        post_count=("vader_compound", "count"),
        positive_pct=("vader_label", lambda x: (x == "positive").mean() * 100),
        negative_pct=("vader_label", lambda x: (x == "negative").mean() * 100)
    ).reset_index()

    if "roberta_numeric" in df.columns:
        roberta_agg = df.groupby("month").agg(
            roberta_mean=("roberta_numeric", "mean"),
            roberta_positive_pct=("roberta_label", lambda x: (x == "positive").mean() * 100),
            roberta_negative_pct=("roberta_label", lambda x: (x == "negative").mean() * 100)
        ).reset_index()
        agg = agg.merge(roberta_agg, on="month")

    agg["month"] = agg["month"].dt.to_timestamp()
    agg["reliable"] = agg["post_count"] >= min_posts
    return agg
