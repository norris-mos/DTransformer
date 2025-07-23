#!/usr/bin/env python3
"""
Generate coldstart dataset for DKT models using diverse cold start test users.

This script creates train/test splits for DKT cold start experiments where:
- Train set: users who never answered cold start questions
- Test set: users who answered at least one cold start question (from diverse_cold_start.json)
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from tqdm import tqdm


class EediColdStartToDTransformer:
    """
    Create cold start datasets for DTransformer/DKT models using pre-selected test users.
    """
    
    def __init__(self, random_seed: int = 42, min_sequence_length: int = 5):
        self.random_seed = random_seed
        self.min_sequence_length = min_sequence_length
        
        # Data storage
        self.answers = None
        self.questions = None
        self.merged_data = None
        self.train_users = None
        self.test_users = None
        self.cold_start_users = set()
        
        # Mappings
        self.question_mapping = None
        self.num_questions = None
        
    def load_cold_start_users(self, cold_start_file: str) -> None:
        """Load cold start test users from JSON file."""
        print(f"Loading cold start test users from {cold_start_file}")
        
        with open(cold_start_file, 'r') as f:
            cold_start_data = json.load(f)
        
        # Get test users from the JSON file
        self.cold_start_users = set(cold_start_data['test_students'])
        
        print(f"Loaded {len(self.cold_start_users)} cold start test users")
        print(f"Coverage: {cold_start_data['coverage_stats']['test_coverage_percentage']:.2f}% of total users")
        
    def load_data(self, data_path: str) -> None:
        """Load EEDI dataset from CSV files."""
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
        
        # Split users based on cold start criteria
        self._split_users_cold_start()
        
        print(f"Data preprocessing complete:")
        print(f"  Total interactions: {len(self.merged_data)}")
        print(f"  Unique questions: {self.num_questions}")
        print(f"  Train users (no cold start questions): {len(self.train_users)}")
        print(f"  Test users (answered cold start questions): {len(self.test_users)}")
        
    def _split_users_cold_start(self) -> None:
        """Split users into train/test based on cold start user list."""
        all_users = set(self.merged_data['UserId'].unique())
        
        # Test users are those in the cold start list
        test_users_in_data = self.cold_start_users.intersection(all_users)
        self.test_users = sorted(list(test_users_in_data))
        
        # Train users are all others
        train_users_in_data = all_users - self.cold_start_users
        self.train_users = sorted(list(train_users_in_data))
        
        print(f"Cold start split:")
        print(f"  Total users in data: {len(all_users)}")
        print(f"  Cold start users found in data: {len(test_users_in_data)} / {len(self.cold_start_users)}")
        print(f"  Train users (never answered cold start questions): {len(self.train_users)}")
        print(f"  Test users (answered cold start questions): {len(self.test_users)}")
        
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
        train_file = os.path.join(output_path, 'train_coldstart.txt')
        
        print(f"Creating cold start training file: {train_file}")
        
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
                
        print(f"Wrote {sequences_written} cold start training sequences")
        
    def create_test_file(self, output_path: str) -> None:
        """Create test.txt file in DTransformer format."""
        test_file = os.path.join(output_path, 'test_coldstart.txt')
        
        print(f"Creating cold start test file: {test_file}")
        
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
                
        print(f"Wrote {sequences_written} cold start test sequences")
        
    def save_metadata(self, output_path: str, cold_start_file: str) -> None:
        """Save metadata about the cold start conversion."""
        metadata_file = os.path.join(output_path, 'coldstart_metadata.txt')
        
        with open(metadata_file, 'w') as f:
            f.write("EEDI Cold Start to DTransformer Conversion Metadata\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Cold start users file: {cold_start_file}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Min sequence length: {self.min_sequence_length}\n\n")
            
            f.write(f"Total unique questions: {self.num_questions}\n")
            f.write(f"Train users (never answered cold start questions): {len(self.train_users)}\n")
            f.write(f"Test users (answered cold start questions): {len(self.test_users)}\n")
            f.write(f"Cold start users provided: {len(self.cold_start_users)}\n\n")
            
            f.write("Question ID mapping (first 20 examples):\n")
            for i, (orig_id, mapped_id) in enumerate(list(self.question_mapping.items())[:20]):
                f.write(f"  {orig_id} -> {mapped_id}\n")
            if len(self.question_mapping) > 20:
                f.write(f"  ... and {len(self.question_mapping) - 20} more\n")
        
        print(f"Saved metadata to {metadata_file}")

    def save_user_lists(self, output_path: str) -> None:
        """Save train/test user lists as JSON files for reference."""
        train_file = os.path.join(output_path, 'coldstart_train_users.json')
        test_file = os.path.join(output_path, 'coldstart_test_users.json')
        
        with open(train_file, 'w') as f:
            json.dump({
                "train_users": [int(user) for user in self.train_users],
                "count": len(self.train_users),
                "description": "Users who never answered cold start questions"
            }, f, indent=2)
        
        with open(test_file, 'w') as f:
            json.dump({
                "test_users": [int(user) for user in self.test_users], 
                "count": len(self.test_users),
                "description": "Users who answered at least one cold start question"
            }, f, indent=2)
        
        print(f"Saved user lists to {train_file} and {test_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate cold start dataset for DTransformer")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to EEDI data directory containing CSV files")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory for coldstart train.txt and test.txt")
    parser.add_argument("--cold_start_users", type=str, required=True,
                       help="Path to JSON file containing cold start test users")
    parser.add_argument("--min_length", type=int, default=5,
                       help="Minimum sequence length to include (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible processing (default: 42)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize converter
    converter = EediColdStartToDTransformer(
        random_seed=args.seed,
        min_sequence_length=args.min_length
    )
    
    # Load cold start users first
    converter.load_cold_start_users(args.cold_start_users)
    
    # Load and process data
    converter.load_data(args.input)
    
    # Create output files
    converter.create_train_file(args.output)
    converter.create_test_file(args.output)
    converter.save_metadata(args.output, args.cold_start_users)
    converter.save_user_lists(args.output)
    
    print("\nCold start conversion complete!")
    print(f"Files created in: {args.output}")
    print("  - train_coldstart.txt")
    print("  - test_coldstart.txt") 
    print("  - coldstart_metadata.txt")
    print("  - coldstart_train_users.json")
    print("  - coldstart_test_users.json")


if __name__ == "__main__":
    main()