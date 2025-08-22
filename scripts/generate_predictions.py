#!/usr/bin/env python3
"""
Generate predictions from a trained DTransformer/DKT model for cold start analysis.

This script loads a trained model and generates predictions on test data,
saving them in the format expected by the cold start comparison scripts.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

# Add the parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tomlkit

from DTransformer.data import KTData
from DTransformer.eval import Evaluator

# Get the DTransformer root directory (parent of scripts directory)
DTRANSFORMER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DTRANSFORMER_ROOT, "data")

def load_model_and_config(model_path, args):
    """Load the trained model and its configuration."""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=args.device)
    
    # Extract configuration from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use provided arguments if no config in checkpoint
        config = {
            'model': args.model,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'n_know': args.n_know,
            'dropout': args.dropout,
            'lambda_cl': args.lambda_cl,
            'proj': args.proj,
            'hard_neg': args.hard_neg,
            'window': args.window
        }
    
    # Get dataset info
    datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
    dataset_config = datasets[args.dataset]
    
    # Initialize model based on type
    if config['model'] == "DKT":
        from baselines.DKT import DKT
        model = DKT(dataset_config["n_questions"], config['d_model'])
    elif config['model'] == "DKVMN":
        from baselines.DKVMN import DKVMN
        model = DKVMN(dataset_config["n_questions"], args.batch_size)
    elif config['model'] == "AKT":
        from baselines.AKT import AKT
        model = AKT(
            dataset_config["n_questions"],
            dataset_config.get("n_pid", 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
        )
    elif config['model'] == "AKT_text":
        from baselines.AKT_text import AKT_text
        model = AKT_text(
            dataset_config["n_questions"],
            dataset_config.get("n_pid", 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
        )
    else:
        # Default to DTransformer
        from DTransformer.model import DTransformer
        model = DTransformer(
            dataset_config["n_questions"],
            dataset_config.get("n_pid", 0),
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_know=config['n_know'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            lambda_cl=config['lambda_cl'],
            proj=config['proj'],
            hard_neg=config['hard_neg'],
            window=config['window'],
        )
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    
    return model, dataset_config, config

def generate_predictions(model, test_data, device, n=1):
    """Generate predictions for all test sequences with T+N look-ahead."""
    all_predictions = []
    all_ground_truth = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating predictions"):
            # Get inputs based on batch structure  
            # Check what fields are available and map accordingly
            if "q" in batch.fields and "s" in batch.fields:
                if "pid" in batch.fields:
                    q, s, pid = batch.get("q", "s", "pid")
                else:
                    q, s = batch.get("q", "s")
                    pid = None
            elif "q" in batch.fields and "r" in batch.fields:
                if "pid" in batch.fields:
                    q, s, pid = batch.get("q", "r", "pid")
                else:
                    q, s = batch.get("q", "r")
                    pid = None
            else:
                raise ValueError(f"Cannot find question and response fields in: {batch.fields}")
            
            # Handle both single sequences and batched sequences (following test.py pattern)
            if not isinstance(q, list):
                q, s, pid = [q], [s], [pid]
            
            for q_seq, s_seq, pid_seq in zip(q, s, pid):
                q_seq = q_seq.to(device)
                s_seq = s_seq.to(device)
                if pid_seq is not None:
                    pid_seq = pid_seq.to(device)
                
                # Use the predict method with look-ahead like in test.py
                y, *_ = model.predict(q_seq, s_seq, pid_seq, n=n)
                
                # Apply sigmoid to get probabilities (like in test.py)
                y_probs = torch.sigmoid(y)
                
                # Get ground truth (shifted by n-1, like in test.py)
                y_true = s_seq[:, (n - 1):]  # Shift ground truth to align with T+N predictions
                
                # Create mask for valid predictions (exclude padding)
                mask = y_true >= 0
                
                # Extract valid predictions and ground truth
                valid_probs = y_probs[mask]
                valid_true = y_true[mask]
                valid_preds = (valid_probs > 0.5).float()
                
                # Store results
                all_ground_truth.extend(valid_true.cpu().numpy())
                all_probs.extend(valid_probs.cpu().numpy())
                all_predictions.extend(valid_preds.cpu().numpy().astype(int))
    
    return np.array(all_ground_truth), np.array(all_predictions), np.array(all_probs)

def parse_test_file_for_users(test_file_path):
    """Parse the test file to extract user sequences and their IDs."""
    user_sequences = []
    user_id = 0
    
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if i + 2 >= len(lines):
            break
            
        seq_len = int(lines[i].strip())
        questions = [int(x) for x in lines[i + 1].strip().split(',')]
        responses = [int(x) for x in lines[i + 2].strip().split(',')]
        
        # Each sequence represents a user
        user_sequences.append({
            'user_id': user_id,
            'seq_len': seq_len,
            'questions': questions,
            'responses': responses
        })
        
        user_id += 1
        i += 3
    
    return user_sequences

def create_predictions_dataframe(ground_truth, predictions, probs, user_sequences, n=1):
    """Create a DataFrame with predictions aligned to users, accounting for T+N look-ahead."""
    
    # Calculate how many predictions each user should have with T+N look-ahead
    user_prediction_counts = []
    for seq in user_sequences:
        # With T+N prediction, we have seq_len - n predictions (we lose n-1 more predictions due to look-ahead)
        prediction_count = max(0, seq['seq_len'] - n)
        user_prediction_counts.append(prediction_count)
    
    # Create user IDs for each prediction
    user_ids = []
    for user_id, count in enumerate(user_prediction_counts):
        user_ids.extend([user_id] * count)
    
    # Ensure we have the right number of predictions
    total_expected = sum(user_prediction_counts)
    actual_predictions = len(ground_truth)
    
    if total_expected != actual_predictions:
        print(f"Warning: Expected {total_expected} predictions but got {actual_predictions}")
        # Truncate or pad as needed
        min_len = min(total_expected, actual_predictions)
        ground_truth = ground_truth[:min_len]
        predictions = predictions[:min_len]
        probs = probs[:min_len]
        user_ids = user_ids[:min_len]
    
    # Create DataFrame
    df = pd.DataFrame({
        'user': user_ids,
        'binary_ground_truth': ground_truth,
        'binary_prediction': predictions,
        'prediction_probs': probs
    })
    
    return df

def main():
    parser = ArgumentParser(description="Generate predictions from trained DTransformer model")
    
    # Model and data
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (must be in datasets.toml)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output CSV file for predictions")
    
    # Device
    parser.add_argument("--device", help="device to run network on", default="cpu")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    
    # Model configuration (used if not saved in checkpoint)
    parser.add_argument("--model", type=str, default="DKT", help="Model type")
    parser.add_argument("--d_model", type=int, default=128, help="model hidden size")
    parser.add_argument("--n_layers", type=int, default=3, help="number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--n_know", type=int, default=32, help="dimension of knowledge parameter")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--lambda_cl", type=float, default=0.1, help="CL loss weight")
    parser.add_argument("--proj", action="store_true", help="projection layer before CL")
    parser.add_argument("--hard_neg", action="store_true", help="use hard negative samples in CL")
    parser.add_argument("--window", type=int, default=1, help="prediction window")
    parser.add_argument("--max_seq_len", type=int, default=None, help="maximum sequence length")
    parser.add_argument("-N", help="T+N prediction window size (look-ahead)", type=int, default=1)
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    
    # Load model and configuration
    model, dataset_config, model_config = load_model_and_config(args.model_path, args)
    
    # Prepare test data
    seq_len = dataset_config.get("seq_len", None)
    if args.max_seq_len is not None:
        seq_len = args.max_seq_len
        print(f"Limiting sequences to max length: {seq_len}")
    
    test_data = KTData(
        os.path.join(DATA_DIR, dataset_config["test"]),
        dataset_config["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
    )
    
    print(f"Loaded {len(test_data)} test sequences")
    
    # Generate predictions
    print(f"Generating predictions with T+{args.N} look-ahead...")
    ground_truth, predictions, probs = generate_predictions(model, test_data, args.device, n=args.N)
    
    # Parse test file to get user information
    test_file_path = os.path.join(DATA_DIR, dataset_config["test"])
    user_sequences = parse_test_file_for_users(test_file_path)
    
    # Create predictions DataFrame
    predictions_df = create_predictions_dataframe(ground_truth, predictions, probs, user_sequences, n=args.N)
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    predictions_df.to_csv(args.output_file, index=False)
    
    print(f"Saved {len(predictions_df)} predictions to: {args.output_file}")
    print(f"Predictions for {predictions_df['user'].nunique()} users")
    
    # Print summary statistics
    accuracy = (predictions_df['binary_ground_truth'] == predictions_df['binary_prediction']).mean()
    print(f"Overall accuracy: {accuracy:.4f}")
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(predictions_df['binary_ground_truth'], predictions_df['prediction_probs'])
    print(f"Overall AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
