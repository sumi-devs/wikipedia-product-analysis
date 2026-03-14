"""
Helper utilities for Wikipedia Top Articles analysis.

Provides functions for:
- API interaction with Wikimedia Pageviews API
- Article classification
- Spike detection and attention decay analysis
- Data aggregation and summarization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Tuple, Optional
import time


class WikimediaAPIClient:
    """Client for Wikimedia Pageviews Top Articles API."""
    
    BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
    
    def __init__(self, project: str = "en.wikipedia.org", access: str = "all-access"):
        """
        Initialize API client.
        
        Args:
            project: Wikimedia project (e.g., en.wikipedia.org)
            access: Access type (all-access, desktop, mobile-app, mobile-web)
        """
        self.project = project
        self.access = access
        self.session = requests.Session()
    
    def fetch_top_articles(self, year: int, month: int, day: int) -> List[Dict]:
        """
        Fetch top 100 articles for a specific day.
        
        Args:
            year, month, day: Date components
            
        Returns:
            List of dicts with keys: article, rank, views, date
        """
        url = f"{self.BASE_URL}/top/{self.project}/{self.access}/{year}/{month:02d}/{day:02d}"
        
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                articles = []
                for idx, item in enumerate(data.get('items', []), 1):
                    articles.append({
                        'date': f"{year}-{month:02d}-{day:02d}",
                        'rank': idx,
                        'article': item.get('article', ''),
                        'views': item.get('views', 0)
                    })
                return articles
            return []
        except Exception as e:
            print(f"Error fetching {year}-{month:02d}-{day:02d}: {e}")
            return []
    
    def fetch_year(self, year: int, start_month: int = 1, end_month: int = 12) -> pd.DataFrame:
        """Fetch top articles for entire year with rate limiting."""
        all_data = []
        days_in_month = {
            1: 31, 2: 29 if year % 4 == 0 else 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        
        total_days = sum(days_in_month[m] for m in range(start_month, end_month + 1))
        processed = 0
        
        for month in range(start_month, end_month + 1):
            for day in range(1, days_in_month[month] + 1):
                articles = self.fetch_top_articles(year, month, day)
                all_data.extend(articles)
                
                processed += 1
                if processed % 30 == 0:
                    print(f"  {processed}/{total_days} days fetched")
                
                time.sleep(0.1)  # Rate limiting
        
        return pd.DataFrame(all_data)


class ArticleClassifier:
    """Classify articles into topic categories."""
    
    CATEGORIES = {
        'Politics': ['election', 'president', 'congress', 'senate', 'parliament', 
                     'government', 'minister', 'vote', 'political', 'democrat', 'republican'],
        'Sports': ['olympic', 'world cup', 'championship', 'super bowl', 'nba', 'nfl',
                   'premier league', 'football', 'soccer', 'baseball', 'tournament'],
        'Entertainment': ['film', 'movie', 'actor', 'actress', 'celebrity', 'music', 'singer',
                          'album', 'award', 'grammy', 'oscar', 'television', 'show', 'tv'],
        'Science': ['physics', 'chemistry', 'biology', 'medicine', 'doctor', 'disease',
                    'scientist', 'research', 'university', 'nobel', 'climate'],
        'Technology': ['google', 'apple', 'microsoft', 'twitter', 'facebook', 'artificial',
                       'ai', 'tech', 'software', 'computer', 'internet', 'crypto']
    }
    
    @classmethod
    def classify(cls, article_title: str) -> str:
        """Classify article based on title keywords."""
        title_lower = article_title.lower()
        for category, keywords in cls.CATEGORIES.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        return 'Other'


class SpikeAnalyzer:
    """Analyze traffic spikes and attention decay."""
    
    @staticmethod
    def calculate_spike_ratio(peak_views: float, avg_views: float) -> float:
        """Calculate spike ratio: peak / average."""
        return peak_views / (avg_views + 1)
    
    @staticmethod
    def detect_spikes(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect articles with spike ratios above threshold.
        
        Args:
            df: DataFrame with 'article', 'views', etc.
            threshold: Spike ratio threshold
            
        Returns:
            DataFrame of spike articles
        """
        article_stats = df.groupby('article').agg({
            'views': ['sum', 'mean', 'max', 'std', 'count']
        }).reset_index()
        
        article_stats.columns = ['article', 'total_views', 'avg_views', 'peak_views', 'std_views', 'count']
        article_stats['std_views'] = article_stats['std_views'].fillna(0)
        article_stats['spike_ratio'] = article_stats['peak_views'] / (article_stats['avg_views'] + 1)
        
        return article_stats[article_stats['spike_ratio'] > threshold].sort_values('spike_ratio', ascending=False)
    
    @staticmethod
    def measure_attention_decay(article_df: pd.DataFrame) -> Dict:
        """
        Measure attention decay after peak traffic.
        
        Args:
            article_df: Time series of views for single article
            
        Returns:
            Dict with decay metrics
        """
        article_df = article_df.sort_values('date')
        peak_idx = article_df['views'].idxmax()
        peak_date = article_df.loc[peak_idx, 'date']
        peak_views = article_df.loc[peak_idx, 'views']
        
        before = article_df[article_df.index < peak_idx]
        after = article_df[article_df.index > peak_idx]
        
        avg_before = before['views'].mean() if len(before) > 0 else 0
        avg_after = after['views'].mean() if len(after) > 0 else 0
        
        decay_pct = ((peak_views - avg_after) / peak_views * 100) if peak_views > 0 else 0
        
        return {
            'peak_date': peak_date,
            'peak_views': peak_views,
            'avg_before': avg_before,
            'avg_after': avg_after,
            'decay_pct': decay_pct,
            'days_to_half': _estimate_half_life(article_df, peak_idx)
        }


def _estimate_half_life(df: pd.DataFrame, peak_idx: int) -> Optional[int]:
    """Estimate days until traffic drops to 50% of peak."""
    post_peak = df.iloc[peak_idx:].reset_index(drop=True)
    peak_views = post_peak.loc[0, 'views']
    half_peak = peak_views / 2
    
    for idx, row in post_peak.iterrows():
        if row['views'] <= half_peak:
            return idx
    return None


def aggregate_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pageviews by day."""
    daily = df.groupby('date').agg({
        'views': ['sum', 'mean', 'max', 'count']
    }).reset_index()
    daily.columns = ['date', 'total_views', 'avg_views', 'max_views', 'articles_count']
    daily['date'] = pd.to_datetime(daily['date'])
    return daily.sort_values('date')


def aggregate_article_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics by article."""
    stats = df.groupby('article').agg({
        'views': ['sum', 'mean', 'median', 'max', 'std', 'count']
    }).reset_index()
    stats.columns = ['article', 'total_views', 'avg_views', 'median_views', 
                     'peak_views', 'std_views', 'appearances']
    stats['std_views'] = stats['std_views'].fillna(0)
    stats['spike_ratio'] = stats['peak_views'] / (stats['avg_views'] + 1)
    return stats.sort_values('total_views', ascending=False)
