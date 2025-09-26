import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import requests
import time

class DataLoader:

    @staticmethod
    def prepare_data(df):
        """
        Prepare data for the trading environment
        """
        # Calculate technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()

        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Forward fill NaN values
        df = df.fillna(method='ffill')

        # Drop any remaining NaN values
        df = df.dropna()

        return df


    @staticmethod
    def load_stock_data(symbol, start_date, end_date, retries=3):
        """Load stock data with multiple fallback options"""
        for attempt in range(retries):
            try:
                print(f"Attempt {attempt + 1} to download data...")

                # Method 1: Try yfinance download
                df = DataLoader._try_yfinance_download(symbol, start_date, end_date)
                if not df.empty:
                    return DataLoader._process_dataframe(df)

                # Method 2: Try yfinance Ticker object
                df = DataLoader._try_yfinance_ticker(symbol, start_date, end_date)
                if not df.empty:
                    return DataLoader._process_dataframe(df)

                # Wait before retry
                time.sleep(2)

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retry
                continue

        print("All download attempts failed. Generating sample data...")
        return DataLoader._process_dataframe(
            DataLoader.generate_sample_data(start_date, end_date)
        )

    @staticmethod
    def _try_yfinance_download(symbol, start_date, end_date):
        """Try downloading using yfinance download method"""
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                show_errors=False
            )
            if not df.empty:
                print("Successfully downloaded data using yfinance download")
                return df
        except Exception as e:
            print(f"yfinance download failed: {str(e)}")
        return pd.DataFrame()

    @staticmethod
    def _try_yfinance_ticker(symbol, start_date, end_date):
        """Try downloading using yfinance Ticker object"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            if not df.empty:
                print("Successfully downloaded data using yfinance Ticker")
                return df
        except Exception as e:
            print(f"yfinance Ticker failed: {str(e)}")
        return pd.DataFrame()

    @staticmethod
    def _process_dataframe(df):
        """Process the dataframe and add technical indicators"""
        try:
            # Ensure required columns exist
            required_columns = ['Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in data")

            # Calculate additional features
            df['Returns'] = df['Close'].pct_change()
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Price_MA'] = df['Close'].rolling(window=20).mean()

            # Add more technical indicators
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()

            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std()

            # Drop NaN values
            df.dropna(inplace=True)

            print(f"Processed dataframe shape: {df.shape}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            print(f"Error processing dataframe: {str(e)}")
            raise

    @staticmethod
    def generate_sample_data(start_date, end_date):
        """Generate realistic sample data for testing"""
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate realistic price movement
        np.random.seed(42)

        # Initial price and parameters
        initial_price = 100
        drift = 0.0001      # Small upward drift
        volatility = 0.02   # Daily volatility

        # Generate prices using geometric Brownian motion
        returns = np.random.normal(drift, volatility, size=len(dates))
        price_factors = np.exp(returns).cumprod()
        prices = initial_price * price_factors

        # Generate volume with realistic patterns
        base_volume = np.random.normal(1000000, 200000, size=len(dates))
        trend_factor = np.linspace(1, 1.5, len(dates))  # Slight upward trend in volume
        volume = base_volume * trend_factor
        volume = np.abs(volume).astype(int)

        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.998, 1.002, size=len(dates)),
            'High': prices * np.random.uniform(1.001, 1.004, size=len(dates)),
            'Low': prices * np.random.uniform(0.996, 0.999, size=len(dates)),
            'Close': prices,
            'Volume': volume,
            'Adj Close': prices  # For compatibility with real data format
        }, index=dates)

        print("Generated sample data successfully")
        return df

    @staticmethod
    def prepare_data2(df):
        """Prepare data for the trading environment"""
        data = []
        try:
            for idx, row in df.iterrows():
                data_point = {
                    'price': float(row['Close']),
                    'volume': float(row['Volume']),
                    'market_feature': float(row['Returns'] if not np.isnan(row['Returns']) else 0.0),
                    'timestamp': idx,
                    'rsi': float(row['RSI'] if 'RSI' in row else 50.0),
                    'volatility': float(row['Volatility'] if 'Volatility' in row else 0.02),
                    'sma_50': float(row['SMA_50'] if 'SMA_50' in row else row['Close']),
                    'sma_200': float(row['SMA_200'] if 'SMA_200' in row else row['Close'])
                }
                data.append(data_point)

            print(f"Prepared {len(data)} data points for trading")
            return data

        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise
