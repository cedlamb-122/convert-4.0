"""
Scikit-learn pipeline for cryptocurrency price prediction using ridge regression.
Contains the main pipeline class for feature engineering, model training, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from loaders import validate_data

class CryptoPricePipeline:
    """
    Machine learning pipeline for cryptocurrency price prediction using ridge regression.
    
    This class handles feature engineering, model training, and evaluation
    for predicting cryptocurrency prices based on timestamp, date features, and volume.
    """
    
    def __init__(self, alpha=1.0, test_size=0.2, random_state=42):
        """
        Initialize the cryptocurrency price prediction pipeline.
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength for Ridge regression
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random state for reproducibility
        """
        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline = None
        self.scaler = None
        
    def _create_features(self, df):
        """
        Create features for the model from the input DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with cryptocurrency data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Convert timestamp to numeric (seconds since epoch)
        features_df['timestamp_numeric'] = features_df['timestamp'].astype(int) // 10**9
        
        # Create lagged features for price and volume
        features_df['price_lag1'] = features_df['price'].shift(1)
        features_df['volume_lag1'] = features_df['volume'].shift(1)
        
        # Create rolling averages
        features_df['price_ma_3'] = features_df['price'].rolling(window=3).mean()
        features_df['volume_ma_3'] = features_df['volume'].rolling(window=3).mean()
        
        # Create price and volume ratios
        features_df['price_volume_ratio'] = features_df['price'] / features_df['volume']
        
        # Drop rows with NaN values created by lagging and rolling operations
        features_df = features_df.dropna()
        
        return features_df
    
    def _prepare_data(self, df):
        """
        Prepare features and target variables for training.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with cryptocurrency data
            
        Returns:
        --------
        tuple
            (X, y) where X is features and y is target
        """
        # Create features
        features_df = self._create_features(df)
        
        # Select feature columns
        feature_columns = [
            'timestamp_numeric', 'volume', 'month', 'day_of_week',
            'price_lag1', 'volume_lag1', 'price_ma_3', 'volume_ma_3',
            'price_volume_ratio'
        ]
        
        X = features_df[feature_columns]
        y = features_df['price']
        
        return X, y
    
    def fit_predict(self, df):
        """
        Fit the model and make predictions, returning evaluation metrics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with cryptocurrency data
            
        Returns:
        --------
        dict
            Dictionary containing train and test metrics (score and MAE)
        """
        # Validate data
        if not validate_data(df):
            raise ValueError("Data validation failed")
        
        # Prepare data
        X, y = self._prepare_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), X.columns.tolist())
            ]
        )
        
        # Create the full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ridge', Ridge(alpha=self.alpha, random_state=self.random_state))
        ])
        
        # Fit the model
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_score': r2_score(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_score': r2_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions on new data using the fitted model.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with cryptocurrency data
            
        Returns:
        --------
        numpy.ndarray
            Predicted prices
        """
        if self.pipeline is None:
            raise ValueError("Model has not been fitted yet. Call fit_predict() first.")
        
        X, _ = self._prepare_data(df)
        return self.pipeline.predict(X)
        