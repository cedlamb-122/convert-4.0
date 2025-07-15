"""
Main script for cryptocurrency price prediction using ridge regression.
This script loads data, creates features, trains a model, and evaluates performance.
"""
import argparse
import os
from pathlib import Path
from sklearn_crypto_pipeline import CryptoPricePipeline
from loaders import load_crypto_data

def main():
    """
    Main function to run the cryptocurrency price prediction pipeline.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction Pipeline')
    parser.add_argument('--data-path', '-d', 
                       type=str, 
                       default='data/crypto_data.csv',
                       help='Path to the cryptocurrency data CSV file (default: data/crypto_data.csv)')
    parser.add_argument('--output-dir', '-o',
                       type=str,
                       default='results',
                       help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file '{data_path}' not found!")
        print("Please check the file path and try again.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load the data
        data = load_crypto_data(str(data_path))
        
        # Create and run the pipeline
        pipeline = CryptoPricePipeline()
        metrics = pipeline.fit_predict(data)
        
        # Print the metrics
        print("\n=== Cryptocurrency Price Prediction Results ===")
        print(f"Train Score (R²): {metrics['train_score']:.4f}")
        print(f"Train MAE: {metrics['train_mae']:.4f}")
        print(f"Test Score (R²): {metrics['test_score']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Save results to file
        results_file = output_dir / 'prediction_results.txt'
        with open(results_file, 'w') as f:
            f.write("=== Cryptocurrency Price Prediction Results ===\n")
            f.write(f"Data file: {data_path}\n")
            f.write(f"Train Score (R²): {metrics['train_score']:.4f}\n")
            f.write(f"Train MAE: {metrics['train_mae']:.4f}\n")
            f.write(f"Test Score (R²): {metrics['test_score']:.4f}\n")
            f.write(f"Test MAE: {metrics['test_mae']:.4f}\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        return

if __name__ == "__main__":
    main()
