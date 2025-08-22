#!/usr/bin/env python3
"""
Standardized Cold Start Dataset Generator for DKT Models

This script creates train/test splits for DKT cold start experiments with the SAME filtering
logic as the LLM preprocessing to ensure identical test sets.

Key standardizations:
- Same minimum sequence length (2 interactions)
- Same maximum sequence length (200 interactions) 
- Same sequence generation logic (one prediction per interaction)
- Same user filtering criteria
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from tqdm import tqdm


class StandardizedEediColdStartToDTransformer:
    """
    Create cold start datasets for DTransformer/DKT models using IDENTICAL filtering to LLM preprocessing.
    """
    
    def __init__(self, random_seed: int = 42, min_sequence_length: int = 2, max_sequence_length: int = 200):
        self.random_seed = random_seed
        self.min_sequence_length = min_sequence_length  # Min: >=2
        self.max_sequence_length = max_sequence_length  # Max: <=200
        
        # No tokenizer needed since we're not doing token-based filtering
        
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
        
        # Print coverage stats if available
        if 'coverage_stats' in cold_start_data:
            print(f"Coverage: {cold_start_data['coverage_stats']['test_coverage_percentage']:.2f}% of total users")
        else:
            print("Coverage stats not available in the cold start file")
        
    def load_data(self, data_path: str) -> None:
        """Load EEDI dataset from CSV files with SAME processing as LLM."""
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
        
        # Clean and merge data - MATCH LLM processing exactly
        print("Merging and cleaning data...")
        
        # Remove unnecessary columns to save memory
        if 'QuizSessionId' in self.answers.columns:
            self.answers = self.answers.drop('QuizSessionId', axis=1)
        
        # Merge answers with questions
        self.merged_data = self.answers.merge(self.questions, on='QuestionId', how='left').dropna().reset_index(drop=True)
        
        # Create question ID mapping using pd.factorize (SAME as LLM)
        self.merged_data['QuestionId_mapped'], self.question_uniques = pd.factorize(self.merged_data['QuestionId'])
        
        # Shift to 1-based indexing (SAME as LLM)
        self.merged_data['QuestionId_mapped'] += 1
        self.question_mapping = {orig_id: mapped_id for orig_id, mapped_id in 
                               zip(self.question_uniques, range(1, len(self.question_uniques) + 1))}
        
        # Update QuestionId to mapped version
        self.merged_data['QuestionId'] = self.merged_data['QuestionId_mapped']
        self.num_questions = len(self.question_uniques)
        
        print(f"Created mapping for {self.num_questions} unique questions")
        
        # Convert IsCorrect to integer (True->1, False->0)
        self.merged_data['Response'] = self.merged_data['IsCorrect'].astype(int)
        
        # Sort by user and date to get chronological order (SAME as LLM)
        self.merged_data = self.merged_data.sort_values(['UserId', 'DateAnswered']).reset_index(drop=True)
        
        # Pre-group data by user for faster access (SAME as LLM)
        self.user_groups = self.merged_data.groupby('UserId')
        
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
        
    def _create_conversation_from_interactions(self, interactions_df: pd.DataFrame) -> list:
        """
        Create conversation format IDENTICAL to LLM preprocessing.
        This replicates the exact logic from eedi2text.py
        """
        conversation = []
        
        for idx, row in interactions_df.iterrows():
            question_id = row['QuestionId']
            is_correct = row['IsCorrect']
            subject_name = row.get('SubjectName', 'Unknown')
            construct_name = row.get('ConstructName', 'Unknown')
            
            # Create user message (question)
            user_content = f"Subject: {subject_name}\nConstruct: {construct_name}\nQuestion ID: {question_id}"
            conversation.append({"role": "user", "content": user_content})
            
            # Create assistant response
            assistant_content = "Correct" if is_correct else "Incorrect"
            conversation.append({"role": "assistant", "content": assistant_content})
        
        return conversation
    
    def _should_include_sequence_token_based(self, interactions_df: pd.DataFrame) -> bool:
        """
        Apply IDENTICAL token-based filtering as LLM preprocessing.
        This replicates the should_include_sequence() function from eedi2text.py
        """
        try:
            # Create conversation format
            conversation = self._create_conversation_from_interactions(interactions_df)
            
            # Apply chat template (SAME as LLM)
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            
            # Tokenize and check length (SAME as LLM)
            tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
            token_length = len(tokens['input_ids'][0])
            
            # Debug: Print token lengths for first few sequences
            if hasattr(self, '_debug_count') and self._debug_count < 5:
                print(f"DEBUG: Sequence length {len(interactions_df)}, tokens: {token_length}, limit: {self.max_token_length}")
                self._debug_count += 1
            elif not hasattr(self, '_debug_count'):
                self._debug_count = 1
                print(f"DEBUG: First sequence - length {len(interactions_df)}, tokens: {token_length}, limit: {self.max_token_length}")
            
            return token_length <= self.max_token_length
            
        except Exception as e:
            print(f"ERROR: Token filtering failed for sequence length {len(interactions_df)}: {e}")
            print(f"DataFrame columns: {list(interactions_df.columns)}")
            print(f"Sample row: {interactions_df.iloc[0].to_dict() if len(interactions_df) > 0 else 'Empty'}")
            return False
    
    def _should_include_user_sequence(self, sorted_interactions: pd.DataFrame) -> bool:
        """
        Apply sequence length filtering (no token-based filtering):
        - Minimum 2 interactions
        - Maximum 200 interactions
        """
        seq_len = len(sorted_interactions)
        
        # Basic length filtering only
        if seq_len < self.min_sequence_length:
            return False
            
        if seq_len > self.max_sequence_length:
            return False
        
        # No token-based filtering for fair comparison
        return True
    
    def create_train_file(self, output_path: str) -> None:
        """Create train.txt file in DTransformer format with SAME filtering as LLM."""
        train_file = os.path.join(output_path, 'train_coldstart_standardized.txt')
        
        print(f"Creating standardized cold start training file: {train_file}")
        
        with open(train_file, 'w') as f:
            sequences_written = 0
            users_filtered_length = 0
            users_filtered_tokens = 0
            
            for user_id in tqdm(self.train_users, desc="Processing train users"):
                if user_id not in self.user_groups.groups:
                    continue
                    
                user_data = self.user_groups.get_group(user_id).sort_values('DateAnswered').reset_index(drop=True)
                
                # Check basic length filtering first
                seq_len = len(user_data)
                # Check length filtering
                if seq_len < self.min_sequence_length or seq_len > self.max_sequence_length:
                    users_filtered_length += 1
                    continue
                
                # No token-based filtering for fair comparison
                    
                problem_ids = user_data['QuestionId'].tolist()
                responses = user_data['Response'].tolist()
                
                # Write in DTransformer format
                f.write(f"{seq_len}\n")
                f.write(",".join(map(str, problem_ids)) + "\n")
                f.write(",".join(map(str, responses)) + "\n")
                
                sequences_written += 1
                
        print(f"Wrote {sequences_written} standardized training sequences")
        print(f"Filtered out {users_filtered_length} users (length criteria)")
        print(f"Filtered out {users_filtered_tokens} users (token criteria - SAME as LLM)")
        
    def create_test_file(self, output_path: str) -> None:
        """Create test.txt file in DTransformer format with SAME progressive filtering as LLM."""
        test_file = os.path.join(output_path, 'test_coldstart_standardized.txt')
        
        print(f"Creating standardized cold start test file: {test_file}")
        print("IMPORTANT: Using PROGRESSIVE filtering - same as LLM (filters each prediction individually)")
        
        with open(test_file, 'w') as f:
            sequences_written = 0
            users_filtered_length = 0
            predictions_filtered_tokens = 0
            total_predictions = 0
            
            for user_id in tqdm(self.test_users, desc="Processing test users"):
                if user_id not in self.user_groups.groups:
                    continue
                    
                user_data = self.user_groups.get_group(user_id).sort_values('DateAnswered').reset_index(drop=True)
                
                # Basic length check (same as LLM)
                seq_len = len(user_data)
                if seq_len < self.min_sequence_length or seq_len > self.max_sequence_length:
                    users_filtered_length += 1
                    continue
                
                # PROGRESSIVE FILTERING: Check each sub-sequence like LLM does
                # LLM creates predictions: 1→2, 1→3, 1→4, ..., 1→n
                # and filters each prediction individually
                valid_predictions = []
                problem_ids = user_data['QuestionId'].tolist()
                responses = user_data['Response'].tolist()
                
                for i in range(1, seq_len):  # For each prediction position
                    # Create sub-sequence up to position i (like LLM progressive)
                    sub_sequence = user_data.iloc[:i+1]  # Include target position
                    
                    # Apply only length filtering (no token filtering for fair comparison)
                    if len(sub_sequence) >= self.min_sequence_length and len(sub_sequence) <= self.max_sequence_length:
                        valid_predictions.append(i)
                    
                    total_predictions += 1
                
                # Only include user if they have valid predictions  
                if valid_predictions:
                    # For DTransformer format, we still write the full sequence
                    # but this ensures only sequences with valid progressive predictions
                    f.write(f"{seq_len}\n")
                    f.write(",".join(map(str, problem_ids)) + "\n")
                    f.write(",".join(map(str, responses)) + "\n")
                    sequences_written += 1
                
        print(f"Wrote {sequences_written} standardized test sequences")
        print(f"Filtered out {users_filtered_length} users (length criteria)")
        print(f"Filtered out {predictions_filtered_tokens}/{total_predictions} individual predictions (token criteria - SAME as LLM)")
        print(f"PROGRESSIVE FILTERING APPLIED: Each prediction filtered individually like LLM")
        
    def save_metadata(self, output_path: str, cold_start_file: str) -> None:
        """Save metadata about the standardized cold start conversion."""
        metadata_file = os.path.join(output_path, 'coldstart_standardized_metadata.txt')
        
        with open(metadata_file, 'w') as f:
            f.write("STANDARDIZED EEDI Cold Start to DTransformer Conversion Metadata\\n")
            f.write("=" * 60 + "\\n\\n")
            f.write("STANDARDIZATION: This uses IDENTICAL filtering to LLM preprocessing\\n")
            f.write("=" * 60 + "\\n\\n")
            f.write(f"Cold start users file: {cold_start_file}\\n")
            f.write(f"Random seed: {self.random_seed}\\n")
            f.write(f"Min sequence length: {self.min_sequence_length} (SAME as LLM)\n")
            f.write(f"Max sequence length: {self.max_sequence_length} (SAME as LLM)\n")
            f.write(f"Max token length: {self.max_token_length} (SAME as LLM)\n")
            f.write(f"Tokenizer: {self.tokenizer.name_or_path} (SAME as LLM)\n\n")
            
            f.write(f"Total unique questions: {self.num_questions}\\n")
            f.write(f"Train users (never answered cold start questions): {len(self.train_users)}\\n")
            f.write(f"Test users (answered cold start questions): {len(self.test_users)}\\n")
            f.write(f"Cold start users provided: {len(self.cold_start_users)}\\n\\n")
            
            f.write("Question ID mapping (first 20 examples):\\n")
            for i, (orig_id, mapped_id) in enumerate(list(self.question_mapping.items())[:20]):
                f.write(f"  {orig_id} -> {mapped_id}\\n")
            if len(self.question_mapping) > 20:
                f.write(f"  ... and {len(self.question_mapping) - 20} more\\n")
        
        print(f"Saved standardized metadata to {metadata_file}")

    def save_user_lists(self, output_path: str) -> None:
        """Save train/test user lists as JSON files for reference."""
        train_file = os.path.join(output_path, 'coldstart_standardized_train_users.json')
        test_file = os.path.join(output_path, 'coldstart_standardized_test_users.json')
        
        with open(train_file, 'w') as f:
            json.dump({
                "train_users": [int(user) for user in self.train_users],
                "count": len(self.train_users),
                "description": "Users who never answered cold start questions (standardized filtering)"
            }, f, indent=2)
        
        with open(test_file, 'w') as f:
            json.dump({
                "test_users": [int(user) for user in self.test_users], 
                "count": len(self.test_users),
                "description": "Users who answered at least one cold start question (standardized filtering)"
            }, f, indent=2)
        
        print(f"Saved standardized user lists to {train_file} and {test_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate STANDARDIZED cold start dataset for DTransformer")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to EEDI data directory containing CSV files")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory for standardized coldstart train.txt and test.txt")
    parser.add_argument("--cold_start_users", type=str, required=True,
                       help="Path to JSON file containing cold start test users")
    parser.add_argument("--min_length", type=int, default=2,
                       help="Minimum sequence length to include (default: 2)")
    parser.add_argument("--max_length", type=int, default=200,
                       help="Maximum sequence length to include (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible processing (default: 42)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize converter with sequence length filtering only
    converter = StandardizedEediColdStartToDTransformer(
        random_seed=args.seed,
        min_sequence_length=args.min_length,
        max_sequence_length=args.max_length
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
    
    print("\nStandardized cold start conversion complete!")
    print(f"Files created in: {args.output}")
    print("  - train_coldstart_standardized.txt")
    print("  - test_coldstart_standardized.txt") 
    print("  - coldstart_standardized_metadata.txt")
    print("  - coldstart_standardized_train_users.json")
    print("  - coldstart_standardized_test_users.json")
    print("\nThese files use sequence length filtering (2-200 interactions) only!")
    print("No token-based filtering applied for fair comparison with LLM preprocessing!")


if __name__ == "__main__":
    main()
