"""
Main script for cryptocurrency price prediction using ridge regression.
Script loads data, creates features, trains a model, and evaluates performance.
"""

from sklearn_crypto_pipeline import CryptoPricePipeline
from loaders import load_crypto_data

def main():
    """
    Main function to run the cryptocurrency price prediction pipeline.
    """
    # Load the data
    data = load_crypto_data('data.csv')
    
    # Create and run the pipeline
    pipeline = CryptoPricePipeline()
    metrics = pipeline.fit_predict(data)
    
    # Print the metrics
    print("=== Cryptocurrency Price Prediction Results ===")
    print(f"Train Score (R²): {metrics['train_score']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.4f}")
    print(f"Test Score (R²): {metrics['test_score']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")

if __name__ == "__main__":
    main()
