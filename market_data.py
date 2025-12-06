"""Additional market data utilities using Alpha Vantage."""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional
import time
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collect additional market data and validate against primary sources."""
    
    def __init__(self, api_key: str):
        """
        Initialize market data collector.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.ti = TechIndicators(key=api_key, output_format='pandas')
    
    def fetch_alpha_vantage_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch daily stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching Alpha Vantage data for {symbol}")
        
        try:
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            
            # Clean column names
            data.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in data.columns]
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'date'}, inplace=True)
            
            # Add symbol
            data['symbol'] = symbol
            
            logger.info(f"Retrieved {len(data)} records for {symbol} from Alpha Vantage")
            
            # Rate limiting (Alpha Vantage free tier: 5 calls/min)
            time.sleep(12)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def validate_data_quality(self, primary_df: pd.DataFrame, 
                             backup_df: pd.DataFrame,
                             symbol: str) -> Dict[str, any]:
        """
        Cross-validate data from multiple sources.
        
        Args:
            primary_df: Primary data source (yFinance)
            backup_df: Backup data source (Alpha Vantage)
            symbol: Stock symbol
            
        Returns:
            Validation report
        """
        logger.info(f"Validating data quality for {symbol}")
        
        validation = {
            'symbol': symbol,
            'primary_records': len(primary_df),
            'backup_records': len(backup_df),
            'date_overlap': 0,
            'price_correlation': None,
            'discrepancies': []
        }
        
        if primary_df.empty or backup_df.empty:
            logger.warning(f"Cannot validate - missing data for {symbol}")
            return validation
        
        # Merge on date
        merged = primary_df.merge(
            backup_df[['date', 'close']], 
            on='date', 
            how='inner',
            suffixes=('_primary', '_backup')
        )
        
        validation['date_overlap'] = len(merged)
        
        if len(merged) > 0:
            # Calculate correlation
            correlation = merged['close_primary'].corr(merged['close_backup'])
            validation['price_correlation'] = correlation
            
            # Find significant discrepancies (>5% difference)
            merged['price_diff_pct'] = abs(
                (merged['close_primary'] - merged['close_backup']) / merged['close_primary'] * 100
            )
            
            discrepancies = merged[merged['price_diff_pct'] > 5]
            if len(discrepancies) > 0:
                validation['discrepancies'] = discrepancies[
                    ['date', 'close_primary', 'close_backup', 'price_diff_pct']
                ].to_dict('records')
                logger.warning(f"Found {len(discrepancies)} price discrepancies for {symbol}")
            
            logger.info(f"Data validation for {symbol}: correlation={correlation:.4f}, "
                       f"overlap={len(merged)} days")
        
        return validation
    
    def fetch_market_indicators(self) -> pd.DataFrame:
        """
        Fetch market-wide indicators (VIX, etc.).
        
        Note: Alpha Vantage doesn't directly provide VIX, but we can fetch SPY
        as a market proxy.
        
        Returns:
            DataFrame with market indicators
        """
        logger.info("Fetching market-wide indicators")
        
        try:
            # Fetch S&P 500 ETF (SPY) as market indicator
            spy_data = self.fetch_alpha_vantage_data('SPY', outputsize='full')
            
            if not spy_data.empty:
                # Calculate market returns
                spy_data['market_return'] = spy_data['close'].pct_change()
                spy_data['market_volatility'] = spy_data['market_return'].rolling(window=20).std()
                
                # Keep only relevant columns
                market_df = spy_data[['date', 'close', 'market_return', 'market_volatility']].copy()
                market_df.columns = ['date', 'spy_close', 'market_return', 'market_volatility']
                
                logger.info(f"Retrieved {len(market_df)} market indicator records")
                return market_df
            
        except Exception as e:
            logger.error(f"Error fetching market indicators: {str(e)}")
        
        return pd.DataFrame()
    
    def reconcile_data(self, primary_df: pd.DataFrame, backup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconcile and fill gaps in primary data using backup source.
        
        Args:
            primary_df: Primary data source
            backup_df: Backup data source
            
        Returns:
            Reconciled DataFrame
        """
        logger.info("Reconciling data from multiple sources")
        
        # Start with primary data
        reconciled = primary_df.copy()
        
        # Find missing dates in primary
        primary_dates = set(primary_df['date'])
        backup_dates = set(backup_df['date'])
        missing_dates = backup_dates - primary_dates
        
        if missing_dates:
            logger.info(f"Filling {len(missing_dates)} missing dates from backup source")
            
            # Get missing records from backup
            missing_records = backup_df[backup_df['date'].isin(missing_dates)]
            
            # Append to primary
            reconciled = pd.concat([reconciled, missing_records], ignore_index=True)
            reconciled = reconciled.sort_values('date')
        
        logger.info(f"Reconciled data: {len(reconciled)} total records")
        
        return reconciled
    
    def get_sector_performance(self, sector_etfs: Dict[str, str]) -> pd.DataFrame:
        """
        Get sector performance using sector ETFs.
        
        Args:
            sector_etfs: Dictionary mapping sector names to ETF symbols
            
        Returns:
            DataFrame with sector performance
        """
        logger.info("Fetching sector performance")
        
        sector_data = []
        
        for sector, etf_symbol in sector_etfs.items():
            try:
                data = self.fetch_alpha_vantage_data(etf_symbol, outputsize='compact')
                if not data.empty:
                    data['sector'] = sector
                    sector_data.append(data)
            except Exception as e:
                logger.error(f"Error fetching {sector} data: {str(e)}")
        
        if sector_data:
            combined = pd.concat(sector_data, ignore_index=True)
            logger.info(f"Retrieved sector data for {len(sector_etfs)} sectors")
            return combined
        
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the collector
    from src.utils import get_api_keys
    
    api_keys = get_api_keys()
    
    collector = MarketDataCollector(api_keys['alpha_vantage'])
    
    # Test fetching data
    print("Fetching Alpha Vantage data for AAPL...")
    df = collector.fetch_alpha_vantage_data('AAPL', outputsize='compact')
    print(df.head())
    
    # Test market indicators
    print("\nFetching market indicators...")
    market_df = collector.fetch_market_indicators()
    print(market_df.head())
