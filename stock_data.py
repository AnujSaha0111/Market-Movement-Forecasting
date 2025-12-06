"""Stock data collection and technical indicator calculation using yFinance."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional
import ta

logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collect stock price data and calculate technical indicators."""
    
    def __init__(self, symbols: List[str], save_path: Optional[Path] = None):
        """
        Initialize the stock data collector.
        
        Args:
            symbols: List of stock symbols to collect data for
            save_path: Path to save collected data
        """
        self.symbols = symbols
        self.save_path = save_path
        self.data = {}
        
    def fetch_stock_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data from yFinance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        logger.info(f"Fetching stock data from {start_date} to {end_date}")
        
        for symbol in self.symbols:
            try:
                logger.info(f"Downloading data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"No data retrieved for {symbol}")
                    continue
                
                # Clean column names
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Reset index to make date a column
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'date'}, inplace=True)
                
                self.data[symbol] = df
                logger.info(f"Retrieved {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return self.data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['open_close_spread'] = (df['close'] - df['open']) / df['open']
        
        # Momentum Indicators
        logger.info("Calculating momentum indicators...")
        
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Rate of Change
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
        
        # Trend Indicators
        logger.info("Calculating trend indicators...")
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # Volatility Indicators
        logger.info("Calculating volatility indicators...")
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_pband'] = bollinger.bollinger_pband()
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Standard Deviation
        df['std_20'] = df['close'].rolling(window=20).std()
        
        # Volume Indicators
        logger.info("Calculating volume indicators...")
        
        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=5)
        
        logger.info(f"Calculated {len(df.columns)} features including technical indicators")
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, target_type: str = 'binary', 
                              threshold: float = 0.0) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: DataFrame with price data
            target_type: 'binary' or 'multiclass'
            threshold: Threshold for classification (percentage)
            
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Calculate next day's return
        df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
        
        if target_type == 'binary':
            # Binary: 1 if price goes up, 0 if down
            df['target'] = (df['next_day_return'] > threshold).astype(int)
            
        elif target_type == 'multiclass':
            # Multiclass: strong_down (-2), down (-1), neutral (0), up (1), strong_up (2)
            conditions = [
                df['next_day_return'] < -0.02,  # Strong down
                (df['next_day_return'] >= -0.02) & (df['next_day_return'] < 0),  # Down
                (df['next_day_return'] >= 0) & (df['next_day_return'] < 0.005),  # Neutral
                (df['next_day_return'] >= 0.005) & (df['next_day_return'] < 0.02),  # Up
                df['next_day_return'] >= 0.02  # Strong up
            ]
            choices = [-2, -1, 0, 1, 2]
            df['target'] = np.select(conditions, choices, default=0)
        
        logger.info(f"Created {target_type} target variable")
        if target_type == 'binary':
            logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def process_all_stocks(self, start_date: str, end_date: str, 
                          target_type: str = 'binary') -> pd.DataFrame:
        """
        Fetch and process data for all stocks.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            target_type: Type of target variable
            
        Returns:
            Combined DataFrame for all stocks
        """
        # Fetch data
        self.fetch_stock_data(start_date, end_date)
        
        # Process each stock
        processed_data = []
        for symbol, df in self.data.items():
            logger.info(f"Processing {symbol}...")
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Create target variable
            df = self.create_target_variable(df, target_type=target_type)
            
            processed_data.append(df)
        
        # Combine all stocks
        combined_df = pd.concat(processed_data, ignore_index=True)
        
        # Save if path provided
        if self.save_path:
            save_file = self.save_path / f"stock_data_{start_date}_{end_date}.csv"
            combined_df.to_csv(save_file, index=False)
            logger.info(f"Saved processed data to {save_file}")
        
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'total_records': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'symbols': df['symbol'].unique().tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        logger.info(f"Data validation: {validation['total_records']} records, "
                   f"{len(validation['symbols'])} symbols")
        
        return validation


if __name__ == "__main__":
    # Test the collector
    from src.utils import load_config, calculate_date_ranges, get_data_path
    
    config = load_config()
    symbols = config['stocks']
    train_start, test_start, test_end = calculate_date_ranges(
        config['data']['training_years'],
        config['data']['testing_months']
    )
    
    collector = StockDataCollector(symbols, save_path=get_data_path('processed'))
    df = collector.process_all_stocks(train_start, test_end)
    
    print("\nData Summary:")
    print(df.info())
    print("\nValidation:")
    print(collector.validate_data(df))
