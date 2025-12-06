"""News data collection from News API and Finnhub."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional
import time
import hashlib
import json

from newsapi import NewsApiClient
import finnhub

logger = logging.getLogger(__name__)


class NewsDataCollector:
    """Collect news headlines from multiple sources."""
    
    def __init__(self, news_api_key: str, finnhub_api_key: str, 
                 cache_path: Optional[Path] = None):
        """
        Initialize news data collector.
        
        Args:
            news_api_key: News API key
            finnhub_api_key: Finnhub API key
            cache_path: Path to cache directory
        """
        self.news_client = NewsApiClient(api_key=news_api_key)
        self.finnhub_client = finnhub.Client(api_key=finnhub_api_key)
        self.cache_path = cache_path
        
        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, source: str) -> str:
        """Generate cache key for news data."""
        key_str = f"{symbol}_{start_date}_{end_date}_{source}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Load news from cache if available."""
        if not self.cache_path:
            return None
        
        cache_file = self.cache_path / f"{cache_key}.json"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_key}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict]) -> None:
        """Save news to cache."""
        if not self.cache_path:
            return
        
        cache_file = self.cache_path / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved to cache: {cache_key}")
    
    def fetch_newsapi_headlines(self, symbol: str, company_name: str,
                                start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch news from News API.
        
        Args:
            symbol: Stock symbol
            company_name: Company name for search
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of news articles
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date, 'newsapi')
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        logger.info(f"Fetching News API headlines for {symbol} ({company_name})")
        
        all_articles = []
        
        try:
            # Search for company name and stock symbol
            query = f"{company_name} OR {symbol}"
            
            # News API has a 1-month limit per request, so we need to batch
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            current_start = start
            while current_start < end:
                current_end = min(current_start + timedelta(days=30), end)
                
                try:
                    response = self.news_client.get_everything(
                        q=query,
                        from_param=current_start.strftime('%Y-%m-%d'),
                        to=current_end.strftime('%Y-%m-%d'),
                        language='en',
                        sort_by='relevancy',
                        page_size=100
                    )
                    
                    if response['status'] == 'ok':
                        articles = response.get('articles', [])
                        all_articles.extend(articles)
                        logger.info(f"Retrieved {len(articles)} articles for {current_start.date()}")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching News API data: {str(e)}")
                
                current_start = current_end
            
            # Save to cache
            self._save_to_cache(cache_key, all_articles)
            
        except Exception as e:
            logger.error(f"Error in News API collection: {str(e)}")
        
        return all_articles
    
    def fetch_finnhub_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch company news from Finnhub.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of news articles
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date, 'finnhub')
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        logger.info(f"Fetching Finnhub news for {symbol}")
        
        all_news = []
        
        try:
            # Finnhub uses Unix timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Fetch company news
            news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            
            if news:
                all_news.extend(news)
                logger.info(f"Retrieved {len(news)} articles from Finnhub")
            
            # Rate limiting
            time.sleep(1)
            
            # Save to cache
            self._save_to_cache(cache_key, all_news)
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub data: {str(e)}")
        
        return all_news
    
    def merge_news_sources(self, newsapi_articles: List[Dict], 
                          finnhub_articles: List[Dict]) -> pd.DataFrame:
        """
        Merge and deduplicate news from multiple sources.
        
        Args:
            newsapi_articles: Articles from News API
            finnhub_articles: Articles from Finnhub
            
        Returns:
            DataFrame with merged news
        """
        merged_news = []
        
        # Process News API articles
        for article in newsapi_articles:
            merged_news.append({
                'date': article.get('publishedAt', '')[:10],  # Extract date
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'source': 'newsapi',
                'url': article.get('url', ''),
                'content': article.get('content', '')
            })
        
        # Process Finnhub articles
        for article in finnhub_articles:
            # Convert Unix timestamp to date
            date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d')
            merged_news.append({
                'date': date,
                'title': article.get('headline', ''),
                'description': article.get('summary', ''),
                'source': 'finnhub',
                'url': article.get('url', ''),
                'content': article.get('summary', '')
            })
        
        # Create DataFrame
        df = pd.DataFrame(merged_news)
        
        if df.empty:
            logger.warning("No news articles found")
            return df
        
        # Remove duplicates based on title similarity
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by date
        df = df.sort_values('date')
        
        logger.info(f"Merged {len(df)} unique articles from both sources")
        
        return df
    
    def clean_headlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess news headlines.
        
        Args:
            df: DataFrame with news data
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove null values
        df = df.dropna(subset=['title'])
        
        # Clean text
        df['title_clean'] = df['title'].str.lower()
        df['title_clean'] = df['title_clean'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Combine title and description for richer text
        df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        df['full_text_clean'] = df['full_text'].str.lower()
        df['full_text_clean'] = df['full_text_clean'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Remove very short articles
        df = df[df['full_text_clean'].str.len() > 20]
        
        logger.info(f"Cleaned {len(df)} articles")
        
        return df
    
    def aggregate_daily_news(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Aggregate news by date.
        
        Args:
            df: DataFrame with news data
            symbol: Stock symbol
            
        Returns:
            DataFrame with daily aggregated news
        """
        if df.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'news_count', 'headlines'])
        
        # Group by date
        daily_news = df.groupby('date').agg({
            'title': lambda x: ' | '.join(x),
            'full_text': lambda x: ' '.join(x),
            'source': 'count'
        }).reset_index()
        
        daily_news.columns = ['date', 'headlines', 'full_text', 'news_count']
        daily_news['symbol'] = symbol
        
        logger.info(f"Aggregated news into {len(daily_news)} daily records")
        
        return daily_news
    
    def collect_news_for_stock(self, symbol: str, company_name: str,
                               start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect and process news for a single stock.
        
        Args:
            symbol: Stock symbol
            company_name: Company name
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with processed news
        """
        logger.info(f"Collecting news for {symbol} ({company_name})")
        
        # Fetch from both sources
        newsapi_articles = self.fetch_newsapi_headlines(symbol, company_name, start_date, end_date)
        finnhub_articles = self.fetch_finnhub_news(symbol, start_date, end_date)
        
        # Merge sources
        df = self.merge_news_sources(newsapi_articles, finnhub_articles)
        
        if df.empty:
            logger.warning(f"No news found for {symbol}")
            return pd.DataFrame()
        
        # Clean headlines
        df = self.clean_headlines(df)
        
        # Aggregate by day
        daily_df = self.aggregate_daily_news(df, symbol)
        
        return daily_df
    
    def collect_news_for_all_stocks(self, symbols: List[str], company_names: Dict[str, str],
                                    start_date: str, end_date: str,
                                    save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Collect news for all stocks.
        
        Args:
            symbols: List of stock symbols
            company_names: Dictionary mapping symbols to company names
            start_date: Start date
            end_date: End date
            save_path: Path to save data
            
        Returns:
            Combined DataFrame with all news
        """
        all_news = []
        
        for symbol in symbols:
            company_name = company_names.get(symbol, symbol)
            df = self.collect_news_for_stock(symbol, company_name, start_date, end_date)
            if not df.empty:
                all_news.append(df)
            
            # Rate limiting between stocks
            time.sleep(2)
        
        # Combine all news
        if all_news:
            combined_df = pd.concat(all_news, ignore_index=True)
            
            # Save if path provided
            if save_path:
                save_file = save_path / f"news_data_{start_date}_{end_date}.csv"
                combined_df.to_csv(save_file, index=False)
                logger.info(f"Saved news data to {save_file}")
            
            return combined_df
        else:
            logger.warning("No news data collected")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the collector
    from src.utils import load_config, get_api_keys, calculate_date_ranges, get_data_path
    
    config = load_config()
    api_keys = get_api_keys()
    
    symbols = config['stocks']
    company_names = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google'
    }
    
    train_start, test_start, test_end = calculate_date_ranges(
        config['data']['training_years'],
        config['data']['testing_months']
    )
    
    collector = NewsDataCollector(
        api_keys['news_api'],
        api_keys['finnhub'],
        cache_path=get_data_path('raw') / 'news_cache'
    )
    
    df = collector.collect_news_for_all_stocks(
        symbols, company_names, train_start, test_end,
        save_path=get_data_path('processed')
    )
    
    print("\nNews Data Summary:")
    print(df.info())
    print("\nSample:")
    print(df.head())
