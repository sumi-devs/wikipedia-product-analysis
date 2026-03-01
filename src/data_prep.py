import pandas as pd

def clean_pageview_data(df):
    """basic cleaning on pageview dataframe."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df = df.dropna()
    df["views"] = df["views"].astype(int)
    return df

def aggregate_data(df, frequency="W"):
    """
    Aggregates pageview data to a given frequency.
    Args:
        df (pd.DataFrame): Dataframe with 'timestamp' and 'views'
        frequency (str): Pandas offset alias (e.g., 'W' for weekly, 'M' for monthly)
    Returns:
        pd.DataFrame: Aggregated data.
    """
    df = df.set_index("timestamp")
    agg_df = df.resample(frequency).sum()
    return agg_df.reset_index()

def add_time_features(df):
    """Adds day of week, month, and year features for seasonality analysis."""
    df = df.copy()
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month_name()
    df["year"] = df["timestamp"].dt.year
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    return df

def clean_reddit_data(df):
    """basic cleaning for reddit data"""
    if df is None or df.empty:
        return df
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def add_pre_post_chatgpt(df, date_col="timestamp"):
    """Tags rows as pre-ChatGPT or post-ChatGPT based on Nov 30, 2022."""
    df = df.copy()
    chatgpt_launch = pd.Timestamp("2022-11-30")
    if date_col in df.columns:
        df["period"] = df[date_col].apply(lambda x: "post-ChatGPT" if pd.notnull(x) and x >= chatgpt_launch else "pre-ChatGPT")
    return df
