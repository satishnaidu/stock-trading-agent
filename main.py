from src.environment.trading_env import TradingEnvironment
from src.agents.ppo_agent import TradingAgent
from src.data.data_loader import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def validate_data(df):
    """Validate the loaded data"""
    if df is None or df.empty:
        return False, "Data is empty"

    required_columns = ['Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        return False, "Missing required columns"

    if len(df) < 100:  # Minimum required length
        return False, "Insufficient data points"

    return True, "Data validation successful"

def main():
    try:
        # Set up dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data

        # Load and prepare data
        data_loader = DataLoader()
        print("Loading stock data...")

        symbols_to_try = ['AAPL', 'MSFT', 'GOOGL']  # Multiple symbols to try
        df = None

        for symbol in symbols_to_try:
            try:
                print(f"\nAttempting to download data for {symbol}...")
                df = data_loader.load_stock_data(symbol, start_date, end_date)
                is_valid, message = validate_data(df)

                if is_valid:
                    print(f"Successfully loaded data for {symbol}")
                    break
                else:
                    print(f"Data validation failed for {symbol}: {message}")
                    df = None
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
                df = None

        if df is None:
            print("Could not load real data, using sample data...")
            df = data_loader.generate_sample_data(start_date, end_date)

        # Validate final dataset
        is_valid, message = validate_data(df)
        if not is_valid:
            raise ValueError(f"Final dataset validation failed: {message}")

        print("\nData Summary:")
        print(f"Shape: {df.shape}")
        print(f"Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"Number of trading days: {len(df)}")

        # Prepare data for trading environment
        print("\nPreparing data for trading environment...")
        prepared_data = data_loader.prepare_data(df)

        # Create environment
        print("Creating trading environment...")
        env = TradingEnvironment(prepared_data)

        # Create and train agent
        print("Creating and training agent...")
        agent = TradingAgent(env)

        # Train the agent
        print("Training agent...")
        agent.train(total_timesteps=10000)

        # Backtest
        print("\nStarting backtest...")
        state, _ = env.reset()
        done = False
        total_reward = 0
        rewards = []

        while not done:
            action = agent.predict(state)
            state, reward, done, _, info = env.step(action)
            total_reward += reward
            rewards.append(reward)

            if len(rewards) % 100 == 0:
                print(f"Step: {len(rewards)}")
                print(f"Average Reward: {np.mean(rewards[-100:]):.2f}")
                print(f"Total P&L: ${info['total_pnl']:.2f}")
                print("-------------------")

        print(f"\nBacktest completed!")
        print(f"Final P&L: ${info['total_pnl']:.2f}")
        print(f"Total Reward: {total_reward:.2f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
