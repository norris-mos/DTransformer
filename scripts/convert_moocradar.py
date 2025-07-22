#!/usr/bin/env python3
"""
Convert MOOC-Radar dataset to DTransformer format.

This script converts the MOOC-Radar educational dataset from its original format
into the simple text format used by DTransformer (sequence length, problem IDs, responses).
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from tqdm import tqdm


class MoocRadarToDTransformer:
    """
    Convert MOOC-Radar dataset to DTransformer format.

    The DTransformer format consists of:
    - Line 1: Sequence length (number of problems)
    - Line 2: Comma-separated problem IDs
    - Line 3: Comma-separated responses (0=incorrect, 1=correct)
    """

    def __init__(self, train_split: float = 0.9, random_seed: int = 42, min_sequence_length: int = 5):
        self.train_split = train_split
        self.random_seed = random_seed
        self.min_sequence_length = min_sequence_length

        # Data storage
        self.data = None
        self.train_users = None
        self.test_users = None

        # Mappings
        self.question_mapping = None
        self.num_questions = None

    def load_data(self, data_path: str) -> None:
        """Load MOOC-Radar dataset."""
        print("Loading MOOC-Radar dataset...")

        # Load JSON file
        data_file = os.path.join(data_path, 'data', 'student-problem-middle.json')
        print(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Flatten nested sequences into individual records
        records = []
        for user_sequence in data:
            for interaction in user_sequence['seq']:
                records.append({
                    'user_id': interaction['user_id'],
                    'problem_id': interaction['problem_id'],
                    'correct': interaction['is_correct'],
                    'timestamp': interaction['submit_time']
                })
        
        self.data = pd.DataFrame(records)
        print(f"Loaded {len(self.data)} interaction records")

        # Create question ID mapping (1-based indexing for DTransformer)
        unique_questions = sorted(self.data['problem_id'].unique())
        self.question_mapping = {qid: idx + 1 for idx, qid in enumerate(unique_questions)}
        self.data['problem_id_mapped'] = self.data['problem_id'].map(self.question_mapping)
        self.num_questions = len(unique_questions)

        print(f"Created mapping for {self.num_questions} unique questions")

        # Convert responses to integer (already 0/1 in MOOC-Radar)
        self.data['response'] = self.data['correct'].astype(int)

        # Convert timestamp to datetime and sort by user and timestamp
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Split users into train/test
        self._split_users()

        print(f"Data preprocessing complete:")
        print(f"  Total interactions: {len(self.data)}")
        print(f"  Unique questions: {self.num_questions}")
        print(f"  Train users: {len(self.train_users)}")
        print(f"  Test users: {len(self.test_users)}")

    def _split_users(self) -> None:
        """Split users into train and test sets."""
        unique_users = self.data['user_id'].unique()

        # Set random seed for reproducible splits
        np.random.seed(self.random_seed)
        shuffled_users = np.random.permutation(unique_users)

        split_idx = int(len(shuffled_users) * self.train_split)
        self.train_users = shuffled_users[:split_idx]
        self.test_users = shuffled_users[split_idx:]

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
        problem_ids = user_data['problem_id_mapped'].tolist()
        responses = user_data['response'].tolist()

        return len(problem_ids), problem_ids, responses

    def create_train_file(self, output_path: str) -> None:
        """Create train.txt file in DTransformer format."""
        train_file = os.path.join(output_path, 'train.txt')

        print(f"Creating training file: {train_file}")

        with open(train_file, 'w') as f:
            sequences_written = 0

            for user_id in tqdm(self.train_users, desc="Processing train users"):
                user_data = self.data[self.data['user_id'] == user_id]

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
                user_data = self.data[self.data['user_id'] == user_id]

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
            f.write("MOOC-Radar to DTransformer Conversion Metadata\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Train split: {self.train_split}\n")
            f.write(f"Min sequence length: {self.min_sequence_length}\n\n")
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
    parser = argparse.ArgumentParser(description="Convert MOOC-Radar dataset to DTransformer format")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to MOOC-Radar data directory containing JSONL files")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory for train.txt and test.txt")
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Fraction of users for training (default: 0.9)")
    parser.add_argument("--min_length", type=int, default=5,
                       help="Minimum sequence length to include (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Initialize converter
    converter = MoocRadarToDTransformer(
        train_split=args.train_split,
        random_seed=args.seed,
        min_sequence_length=args.min_length
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
