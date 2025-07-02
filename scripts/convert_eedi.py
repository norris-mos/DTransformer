#!/usr/bin/env python3
"""
Convert EEDI dataset to DTransformer format.

This script converts the EEDI educational dataset from its original CSV format
to the simple text format used by DTransformer (sequence length, problem IDs, responses).
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from tqdm import tqdm


class EediToDTransformer:
    """
    Convert EEDI dataset to DTransformer format.
    
    The DTransformer format consists of:
    - Line 1: Sequence length (number of problems)
    - Line 2: Comma-separated problem IDs
    - Line 3: Comma-separated responses (0=incorrect, 1=correct)
    """
    
    def __init__(self, train_split: float = 0.9, random_seed: int = 42, min_sequence_length: int = 5, filtered_users_file: str = None):
        self.train_split = train_split
        self.random_seed = random_seed
        self.min_sequence_length = min_sequence_length
        self.filtered_users_file = filtered_users_file
        
        # Data storage
        self.answers = None
        self.questions = None
        self.merged_data = None
        self.train_users = None
        self.test_users = None
        self.filtered_users = set()  # Track users to exclude
        
        # Mappings
        self.question_mapping = None
        self.num_questions = None
        
    def load_filtered_users(self) -> None:
        """Load the list of filtered users to exclude from processing."""
        if self.filtered_users_file and os.path.exists(self.filtered_users_file):
            print(f"Loading filtered users from {self.filtered_users_file}")
            with open(self.filtered_users_file, 'r') as f:
                filtered_data = json.load(f)
                # Convert to set of integers for faster lookup
                self.filtered_users = set(int(user) for user in filtered_data.get('filtered_users', []))
            print(f"Loaded {len(self.filtered_users)} filtered users")
        else:
            print("No filtered users file provided or file doesn't exist")
    
    def load_data(self, data_path: str) -> None:
        """Load EEDI dataset from CSV files."""
        # Load filtered users first
        self.load_filtered_users()
        
        print("Loading EEDI dataset...")
        
        # Load CSV files
        answers_path = os.path.join(data_path, 'answer.csv')
        questions_path = os.path.join(data_path, 'questions.csv')
        
        print(f"Loading answers from {answers_path}")
        self.answers = pd.read_csv(answers_path)
        print(f"Loaded {len(self.answers)} answer records")
        
        print(f"Loading questions from {questions_path}")
        self.questions = pd.read_csv(questions_path)
        print(f"Loaded {len(self.questions)} question records")
        
        # Clean and merge data
        print("Merging and cleaning data...")
        
        # Remove unnecessary columns to save memory
        if 'QuizSessionId' in self.answers.columns:
            self.answers = self.answers.drop('QuizSessionId', axis=1)
        
        # Merge answers with questions (we only need QuestionId and IsCorrect)
        self.merged_data = self.answers[['UserId', 'QuestionId', 'IsCorrect', 'DateAnswered']].copy()
        
        # Remove any rows with missing data
        initial_len = len(self.merged_data)
        self.merged_data = self.merged_data.dropna().reset_index(drop=True)
        print(f"Removed {initial_len - len(self.merged_data)} rows with missing data")
        
        # Create question ID mapping (1-based indexing for DTransformer)
        unique_questions = sorted(self.merged_data['QuestionId'].unique())
        self.question_mapping = {qid: idx + 1 for idx, qid in enumerate(unique_questions)}
        self.merged_data['QuestionId_mapped'] = self.merged_data['QuestionId'].map(self.question_mapping)
        self.num_questions = len(unique_questions)
        
        print(f"Created mapping for {self.num_questions} unique questions")
        
        # Convert IsCorrect to integer (True->1, False->0)
        self.merged_data['Response'] = self.merged_data['IsCorrect'].astype(int)
        
        # Sort by user and date to get chronological order
        self.merged_data = self.merged_data.sort_values(['UserId', 'DateAnswered']).reset_index(drop=True)
        
        # Split users into train/test
        self._split_users()
        
        print(f"Data preprocessing complete:")
        print(f"  Total interactions: {len(self.merged_data)}")
        print(f"  Unique questions: {self.num_questions}")
        print(f"  Train users: {len(self.train_users)}")
        print(f"  Test users: {len(self.test_users)}")
        
    def _split_users(self) -> None:
        """Split users into train and test sets, then filter out excluded users."""
        unique_users = self.merged_data['UserId'].unique()
        
        # Set random seed for reproducible splits
        np.random.seed(self.random_seed)
        shuffled_users = np.random.permutation(unique_users)
        
        split_idx = int(len(shuffled_users) * self.train_split)
        train_users = shuffled_users[:split_idx]
        test_users = shuffled_users[split_idx:]
        
        # Filter out excluded users AFTER splitting to maintain consistency with eedi2text.py
        if self.filtered_users:
            initial_train = len(train_users)
            initial_test = len(test_users)
            
            self.train_users = [user for user in train_users if user not in self.filtered_users]
            self.test_users = [user for user in test_users if user not in self.filtered_users]
            
            print(f"Filtered out {initial_train - len(self.train_users)} train users")
            print(f"Filtered out {initial_test - len(self.test_users)} test users")
        else:
            self.train_users = train_users
            self.test_users = test_users
        
        print(f"Final split: {len(self.train_users)} train, {len(self.test_users)} test users")
        
    def _process_user_sequence(self, user_data: pd.DataFrame) -> tuple:
        """
        Process a single user's data into DTransformer format.
        
        Returns:
            tuple: (sequence_length, problem_ids, responses) or None if too short
        """
        if len(user_data) < self.min_sequence_length:
            return None
            
        # Extract sequences
        problem_ids = user_data['QuestionId_mapped'].tolist()
        responses = user_data['Response'].tolist()
        
        return len(problem_ids), problem_ids, responses
    
    def create_train_file(self, output_path: str) -> None:
        """Create train.txt file in DTransformer format."""
        train_file = os.path.join(output_path, 'train.txt')
        
        print(f"Creating training file: {train_file}")
        
        with open(train_file, 'w') as f:
            sequences_written = 0
            
            for user_id in tqdm(self.train_users, desc="Processing train users"):
                user_data = self.merged_data[self.merged_data['UserId'] == user_id]
                
                result = self._process_user_sequence(user_data)
                if result is None:
                    continue
                    
                seq_len, problem_ids, responses = result
                
                # Write in DTransformer format
                f.write(f"{seq_len}\n")
                f.write(",".join(map(str, problem_ids)) + "\n")
                f.write(",".join(map(str, responses)) + "\n")
                
                sequences_written += 1
                
        print(f"Wrote {sequences_written} training sequences")
        
    def create_test_file(self, output_path: str) -> None:
        """Create test.txt file in DTransformer format."""
        test_file = os.path.join(output_path, 'test.txt')
        
        print(f"Creating test file: {test_file}")
        
        with open(test_file, 'w') as f:
            sequences_written = 0
            
            for user_id in tqdm(self.test_users, desc="Processing test users"):
                user_data = self.merged_data[self.merged_data['UserId'] == user_id]
                
                result = self._process_user_sequence(user_data)
                if result is None:
                    continue
                    
                seq_len, problem_ids, responses = result
                
                # Write in DTransformer format
                f.write(f"{seq_len}\n")
                f.write(",".join(map(str, problem_ids)) + "\n")
                f.write(",".join(map(str, responses)) + "\n")
                
                sequences_written += 1
                
        print(f"Wrote {sequences_written} test sequences")
        
    def save_metadata(self, output_path: str) -> None:
        """Save metadata about the conversion."""
        metadata_file = os.path.join(output_path, 'metadata.txt')
        
        with open(metadata_file, 'w') as f:
            f.write("EEDI to DTransformer Conversion Metadata\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Train split: {self.train_split}\n")
            f.write(f"Min sequence length: {self.min_sequence_length}\n")
            f.write(f"Filtered users file: {self.filtered_users_file}\n")
            f.write(f"Number of filtered users: {len(self.filtered_users)}\n\n")
            f.write(f"Total unique questions: {self.num_questions}\n")
            f.write(f"Train users: {len(self.train_users)}\n")
            f.write(f"Test users: {len(self.test_users)}\n\n")
            f.write("Question ID mapping (original -> mapped):\n")
            
            # Save first 20 mappings as examples
            for i, (orig_id, mapped_id) in enumerate(list(self.question_mapping.items())[:20]):
                f.write(f"  {orig_id} -> {mapped_id}\n")
            if len(self.question_mapping) > 20:
                f.write(f"  ... and {len(self.question_mapping) - 20} more\n")
        
        print(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert EEDI dataset to DTransformer format")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to EEDI data directory containing CSV files")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory for train.txt and test.txt")
    parser.add_argument("--train_split", type=float, default=0.9,  # Changed from 0.8 to match eedi2text
                       help="Fraction of users for training (default: 0.9)")
    parser.add_argument("--min_length", type=int, default=5,
                       help="Minimum sequence length to include (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--filtered_users", type=str, default=None,
                       help="Path to JSON file containing filtered users to exclude")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize converter
    converter = EediToDTransformer(
        train_split=args.train_split,
        random_seed=args.seed,
        min_sequence_length=args.min_length,
        filtered_users_file=args.filtered_users
    )
    
    # Load and process data
    converter.load_data(args.input)
    
    # Create output files
    converter.create_train_file(args.output)
    converter.create_test_file(args.output)
    converter.save_metadata(args.output)
    
    print("\nConversion complete!")
    print(f"Files created in: {args.output}")
    print("  - train.txt")
    print("  - test.txt")
    print("  - metadata.txt")


if __name__ == "__main__":
    main()
