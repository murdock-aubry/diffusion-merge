from datasets import load_dataset
import json
import os
from typing import List, Dict, Any
import re
from huggingface_hub import list_repo_files
import datasets.utils.info_utils as info_utils
from datetime import datetime, date
import pickle
import pandas as pd


class DatasetSharder:
    def __init__(self, categories: List[str], threshold: float = 0.5):
        """
        Initialize the dataset sharder with categories and classification threshold.
        
        :param categories: List of categories to classify prompts into
        :param threshold: Minimum confidence score to assign a prompt to a category
        """
        self.categories = categories
        self.threshold = threshold
        
    def _create_category_classifiers(self) -> Dict[str, List[str]]:
        """
        Create keyword lists for each category to help with classification.
        
        :return: Dictionary of categories with associated keyword lists
        """
        category_keywords = {
            "body-parts": [
                "hand", "foot", "leg", "arm", "finger", "toe", 
                "head", "face", "eye", "nose", "ear", "mouth"
            ],
            "electronics": [
                "phone", "laptop", "computer", "tablet", "smartwatch", 
                "drone", "camera", "headphones", "speaker", "tv", 
                "monitor", "keyboard", "mouse", "charger", "microphone", 
                "printer", "router", "modem", "game console", "smart home", 
                "robot", "sensor", "drone", "circuit", "hardware"
            ],
            "clothes": [
                "shirt", "pants", "dress", "jacket", "coat", "sweater", 
                "hat", "shoes", "socks", "underwear", "belt", "gloves"
            ],
            "vehicles": [
                "car", "truck", "bicycle", "motorcycle", "bus", 
                "train", "airplane", "helicopter", "boat", "ship", 
                "scooter", "van", "suv"
            ],
            "animals": [
                "dog", "cat", "bird", "horse", "cow", "sheep", 
                "pig", "chicken", "fish", "elephant", "lion", 
                "tiger", "bear", "rabbit", "mouse", "snake"
            ],
            "food": [
                "pizza", "burger", "salad", "cake", "sandwich", "sushi", 
                "pasta", "steak", "soup", "bread", "fruit", "vegetable", 
                "chocolate", "ice cream", "apple", "banana", "orange", 
                "chicken", "fish", "rice", "noodles", "cheese"
            ],
            "text": [
                "write", "text", "sign", "label", "logo", "typography", 
                "font", "quote", "poem", "letter", "alphabet", "word", 
                "manuscript", "handwriting", "calligraphy", "message"
            ]
        }
        
        # Add fallback in case of custom categories
        for category in self.categories:
            if category not in category_keywords:
                category_keywords[category] = []
        
        return category_keywords
    
    def classify_prompt(self, prompt: str) -> Dict[str, float]:
        """
        Classify a prompt into categories based on keyword matching.
        
        :param prompt: Input prompt to classify
        :return: Dictionary of category scores
        """
        # Lowercase the prompt for case-insensitive matching
        prompt_lower = prompt.lower()
        
        # Get keyword classifiers
        category_keywords = self._create_category_classifiers()
        
        # Calculate category scores
        category_scores = {}
        for category, keywords in category_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in prompt_lower)
            
            # Calculate score based on matches
            score = matches / len(keywords) if keywords else 0
            category_scores[category] = score
        
        return category_scores
    
    def categorize_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize entire dataset into specialist shards.
        
        :param dataset: List of prompt-image pair dictionaries
        :return: Dictionary of categorized datasets
        """
        # Initialize output shards
        shards = {category: [] for category in self.categories}
        uncategorized = []
        
        # Categorize each item
        for item in dataset:
            # Create a copy to avoid modifying the original dataset
            item_copy = item.copy()
            
            prompt = item_copy.get('caption', '')
            scores = self.classify_prompt(prompt)
            
            # Find categories above threshold
            matched_categories = [
                cat for cat, score in scores.items() 
                if score > self.threshold
            ]

            
            # Allocate the sample to each category with a nonzero score
            for category, score in scores.items():
                if score > 0:
                    shards[category].append({'item': item_copy, 'score': score})
        # Filter each category to only include the top M items based on score

        M = 1111 #results in 1000 train, 100 test

        for category in shards.keys():
            sorted_items = sorted(shards[category], key=lambda x: x['score'], reverse=True)[:M]
            train_split = sorted_items[:int(M * 0.9)]
            test_split = sorted_items[int(M * 0.9):]
            shards[category] = {'train': train_split, 'test': test_split}
        
        # Optional: Add uncategorized to a separate shard
        shards['uncategorized'] = uncategorized

        return shards
    
    def save_shards(self, shards: Dict[str, Dict[str, List[Dict[str, Any]]]], output_dir: str, file_type: str = 'parquet'):
        """
        Save categorized shards to files, excluding the "uncategorized" shard.
        
        :param shards: Categorized dataset shards with nested structure
        :param output_dir: Directory to save shard files
        :param file_type: File type to save ('pickle' or 'parquet')
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each shard, excluding "uncategorized"
        for category, split_data in shards.items():
            if category == 'uncategorized':
                continue  # Skip saving the "uncategorized" shard
            
            # Handle each split (train, test, etc.)
            for split, data in split_data.items():
                # Create subdirectory for the split if needed
                split_dir = os.path.join(output_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                output_path = os.path.join(split_dir, f"{category}_shard")
                
                if file_type == 'pickle':
                    # Save using pickle (preserves most Python objects)
                    output_path += '.pkl'
                    with open(output_path, 'wb') as f:
                        pickle.dump(data, f)
                
                elif file_type == 'parquet':
                    # Save using parquet (efficient for large datasets)
                    output_path += '.parquet'
                    # Convert to DataFrame first
                    df = pd.DataFrame(data)
                    df.to_parquet(output_path, index=False)
                
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
        
        # Print shard sizes, excluding "uncategorized"
        print("Shard Sizes:")
        for category, split_data in shards.items():
            if category == 'uncategorized':
                continue  # Skip printing the "uncategorized" shard size
            for split, data in split_data.items():
                print(f"{category} ({split}): {len(data)} items")


# Example usage
def main():

    data_name = "yuvalkirstain/pickapic_v1"

    N = 22
    all_files = list_repo_files(data_name, repo_type="dataset")

    # Filter and sort files
    train_files = sorted([f for f in all_files if f.startswith("data/train-")])[:N]
    test_files = sorted([f for f in all_files if f.startswith("data/test-")])[:1]
    validation_files = sorted([f for f in all_files if f.startswith("data/validation-")])[:1]
    validation_unique_files = sorted([f for f in all_files if f.startswith("data/validation_unique-")])[:1]
    test_unique_files = sorted([f for f in all_files if f.startswith("data/test_unique-")])[:1]

    # Load dataset
    dataset = load_dataset(
        data_name,
        data_files={
            "train": train_files,
            "test": test_files,
            "validation": validation_files,
            "validation_unique": validation_unique_files,
            "test_unique": test_unique_files
        },
        verification_mode='no_checks'
    )

    dataset = dataset["train"]

 
    # Categories to classify
    categories = ["body-parts", "electronics", "clothes", "vehicles", "animals", "food", "text"]
    
    # Initialize sharder
    sharder = DatasetSharder(categories, threshold=0.0)
    
    # Categorize dataset
    shards = sharder.categorize_dataset(dataset)

    # Save shards

    data_path = "/projects/dynamics/diffusion-tmp/data"

    sharder.save_shards(shards, data_path)

if __name__ == "__main__":
    main()