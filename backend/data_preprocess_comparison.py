#!/usr/bin/env python3
"""
Data preprocessing script for comparison files.
Converts comparison data to format: post, summary, pos_ex, neg_ex
Removes duplicate entries based on post content.
Splits data into train and validation sets.
"""

import json
import os
import argparse
from typing import Dict, List, Set
from pathlib import Path


def load_comparison_file(filepath: str) -> List[Dict]:
    """Load and parse a comparison JSON file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {filepath}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    return data


def process_comparison_entry(entry: Dict) -> Dict:
    """Process a single comparison entry into the desired format."""
    try:
        # Extract post content
        post_content = entry['info'].get('post', '')
        
        # Extract summaries
        summaries = entry.get('summaries', [])
        if len(summaries) != 2:
            print(f"Warning: Entry has {len(summaries)} summaries instead of 2")
            return None
        
        # Determine which summary was chosen
        choice = entry.get('choice', 0)
        
        # Get positive (chosen) and negative (not chosen) examples
        pos_ex = summaries[choice]['text']
        neg_ex = summaries[1 - choice]['text']
        
        # For summary field, we'll use the positive example
        summary = pos_ex
        
        # Get the split information
        split = entry.get('split', 'train')  # Default to train if not specified
        
        return {
            'post': post_content,
            'summary': summary,
            'pos_ex': pos_ex,
            'neg_ex': neg_ex,
            'split': split
        }
    
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Error processing entry: {e}")
        return None


def remove_duplicates(data: List[Dict]) -> List[Dict]:
    """Remove duplicate entries based on post content."""
    seen_posts: Set[str] = set()
    unique_data = []
    
    for entry in data:
        post_content = entry['post']
        
        # Create a hash of the post content to check for duplicates
        post_hash = hash(post_content)
        
        if post_hash not in seen_posts:
            seen_posts.add(post_hash)
            unique_data.append(entry)
    
    return unique_data


def split_data_by_split_field(data: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets based on the split field."""
    train_data = []
    valid_data = []
    
    for entry in data:
        split = entry.get('split', 'train')
        
        # Remove the split field from the final output
        output_entry = {
            'post': entry['post'],
            'summary': entry['summary'],
            'pos_ex': entry['pos_ex'],
            'neg_ex': entry['neg_ex']
        }
        
        if split == 'train':
            train_data.append(output_entry)
        else:  # valid, valid1, valid2, etc.
            valid_data.append(output_entry)
    
    return train_data, valid_data


def save_data(data: List[Dict], output_file: str):
    """Save data to a JSONL file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved {len(data)} entries to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")


def process_all_comparison_files(input_dir: str, output_train: str, output_valid: str):
    """Process all comparison files in the input directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    all_processed_data = []
    
    # Process each JSON file in the directory
    for file_path in input_path.glob('*.json'):
        print(f"Processing {file_path.name}...")
        
        # Load comparison data
        comparison_data = load_comparison_file(str(file_path))
        
        if not comparison_data:
            print(f"Warning: No data loaded from {file_path.name}")
            continue
        
        # Process each entry
        processed_count = 0
        for entry in comparison_data:
            processed_entry = process_comparison_entry(entry)
            if processed_entry:
                all_processed_data.append(processed_entry)
                processed_count += 1
        
        print(f"Processed {processed_count} entries from {file_path.name}")
    
    print(f"\nTotal entries before deduplication: {len(all_processed_data)}")
    
    # Remove duplicates
    unique_data = remove_duplicates(all_processed_data)
    
    print(f"Total entries after deduplication: {len(unique_data)}")
    print(f"Removed {len(all_processed_data) - len(unique_data)} duplicate entries")
    
    # Split data into train and validation sets
    train_data, valid_data = split_data_by_split_field(unique_data)
    
    print(f"\nData split summary:")
    print(f"Training entries: {len(train_data)}")
    print(f"Validation entries: {len(valid_data)}")
    
    # Save the split data
    save_data(train_data, output_train)
    save_data(valid_data, output_valid)


def main():
    parser = argparse.ArgumentParser(description='Preprocess comparison data files')
    parser.add_argument('--input_dir', '-i', default='data/comparisons', 
                       help='Input directory containing comparison JSON files')
    parser.add_argument('--output_train', '-t', default='comparisons_train.jsonl',
                       help='Output file for training data')
    parser.add_argument('--output_valid', '-v', default='comparisons_valid.jsonl',
                       help='Output file for validation data')
    
    args = parser.parse_args()
    
    print("Starting comparison data preprocessing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Training output: {args.output_train}")
    print(f"Validation output: {args.output_valid}")
    
    process_all_comparison_files(args.input_dir, args.output_train, args.output_valid)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main() 