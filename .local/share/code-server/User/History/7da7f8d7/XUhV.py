"""
Data loading utilities for cryptocurrency price prediction.
Handles loading and preprocessing of cryptocurrency data from CSV files.
"""

import pandas as pd  # pylint: disable=import-error

def load_crypto_data(file_path):
    """
    Load cryptocurrency data from CSV file and perform initial preprocessing.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing cryptocurrency data

    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame with timestamp, date features, price, and volume
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Drop the market_cap column as specified
    if 'market_cap' in df.columns:
        df = df.drop('market_cap', axis=1)

    # Convert timestamp to datetime if it's in milliseconds
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Convert date column to datetime if it's not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Extract month and day of week from date
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Sort by timestamp to ensure proper ordering
    df = df.sort_values('timestamp')

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} records from {file_path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df

def validate_data(df):
    """
    Validate that the DataFrame contains required columns and no missing values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate

    Returns:
    --------
    bool
        True if data is valid, False otherwise
    """
    required_columns = ['timestamp', 'price', 'volume', 'month', 'day_of_week']

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False

    # Check for missing values
    if df[required_columns].isnull().any().any():
        print("Data contains missing values")
        return False

    return True
