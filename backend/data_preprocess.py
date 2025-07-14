import json
import os
from pathlib import Path
from typing import Dict, List, Any

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON file and return its contents as a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def extract_opensc_data(data_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract OpenSC dataset from the data directory.
    
    Returns:
        Dictionary with 'train' and 'valid' keys, each containing a list of
        dictionaries with 'prompt' and 'ideal_summary' fields.
    """
    train_data = {}
    valid_data = {}
    
    # Process axis_evals directory
    axis_evals_dir = Path(data_dir) / "axis_evals"
    if axis_evals_dir.exists():
        for json_file in axis_evals_dir.glob("*.json"):
            print(f"Processing {json_file}...")
            try:
                data = load_json_file(str(json_file))
                for item in data:
                    if "info" in item and "summaries" in item:
                        # Handle different possible structures
                        prompt = None
                        if "post" in item["info"]:
                            prompt = item["info"]["post"]
                        elif "text" in item["info"]:
                            prompt = item["info"]["text"]
                        elif "article" in item["info"]:
                            prompt = item["info"]["article"]
                        
                        if not prompt:
                            continue
                        
                        # Find the best summary (prefer human_editor, then ref, then first available)
                        ideal_summary = None
                        summaries = item["summaries"]
                        
                        # First try to find human_editor
                        for summary in summaries:
                            if summary.get("policy") == "human_editor":
                                ideal_summary = summary["text"]
                                break
                        
                        # If no human_editor, try ref
                        if ideal_summary is None:
                            for summary in summaries:
                                if summary.get("policy") == "ref":
                                    ideal_summary = summary["text"]
                                    break
                        
                        # If still no ideal summary, use the first one
                        if ideal_summary is None and summaries:
                            ideal_summary = summaries[0]["text"]
                        
                        if prompt and ideal_summary:
                            entry = {
                                "prompt": prompt,
                                "ideal_summary": ideal_summary
                            }
                            
                            # Use full prompt as key to avoid duplicates
                            prompt_key = prompt
                            
                            # Determine split based on the split field
                            split = item.get("split", "")
                            if split.startswith("valid"):
                                valid_data[prompt_key] = entry
                            else:
                                train_data[prompt_key] = entry
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    # Process comparisons directory
    comparisons_dir = Path(data_dir) / "comparisons"
    if comparisons_dir.exists():
        for json_file in comparisons_dir.glob("*.json"):
            print(f"Processing {json_file}...")
            try:
                data = load_json_file(str(json_file))
                for item in data:
                    if "info" in item and "summaries" in item:
                        # Handle different possible structures
                        prompt = None
                        if "post" in item["info"]:
                            prompt = item["info"]["post"]
                        elif "text" in item["info"]:
                            prompt = item["info"]["text"]
                        elif "article" in item["info"]:
                            prompt = item["info"]["article"]
                        
                        if not prompt:
                            continue
                        
                        # Find the best summary (prefer human_editor, then ref, then first available)
                        ideal_summary = None
                        summaries = item["summaries"]
                        
                        # First try to find human_editor
                        for summary in summaries:
                            if summary.get("policy") == "human_editor":
                                ideal_summary = summary["text"]
                                break
                        
                        # If no human_editor, try ref
                        if ideal_summary is None:
                            for summary in summaries:
                                if summary.get("policy") == "ref":
                                    ideal_summary = summary["text"]
                                    break
                        
                        # If still no ideal summary, use the first one
                        if ideal_summary is None and summaries:
                            ideal_summary = summaries[0]["text"]
                        
                        if prompt and ideal_summary:
                            entry = {
                                "prompt": prompt,
                                "ideal_summary": ideal_summary
                            }
                            
                            # Use full prompt as key to avoid duplicates
                            prompt_key = prompt
                            
                            # Determine split based on the split field
                            split = item.get("split", "")
                            if split.startswith("valid"):
                                valid_data[prompt_key] = entry
                            else:
                                train_data[prompt_key] = entry
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    return {
        "train": list(train_data.values()),
        "valid": list(valid_data.values())
    }

def save_jsonl(data: List[Dict[str, str]], output_path: str):
    """Save data to JSONL format."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    """Main function to extract and save the OpenSC dataset."""
    data_dir = "../data"
    
    print("Extracting OpenSC dataset...")
    dataset = extract_opensc_data(data_dir)
    
    print(f"Found {len(dataset['train'])} unique training examples")
    print(f"Found {len(dataset['valid'])} unique validation examples")
    
    # Save train data
    train_path = "../data/train.jsonl"
    save_jsonl(dataset["train"], train_path)
    print(f"Saved training data to {train_path}")
    
    # Save validation data
    valid_path = "../data/valid.jsonl"
    save_jsonl(dataset["valid"], valid_path)
    print(f"Saved validation data to {valid_path}")

if __name__ == "__main__":
    main() 