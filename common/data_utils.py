#!/usr/bin/env python3
"""
Data loading utilities for Grammar Search experiments.
Consolidated module to handle MATH, GPQA, AIME, and MuSiQue datasets with proper caching.
"""

import json
import re
import os
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset

# Fixed validation seed for reproducibility
VALIDATION_SEED = 42

# Fixed test seed where test set is too big
TEST_SEED = 123

# Cache directory for validation sets
VALIDATION_CACHE_DIR = Path("validation_set_cache")

# Cache directory for test sets (separate from validation)
TEST_CACHE_DIR = Path("test_set_cache")

# AIME year split configuration
AIME_TEST_YEARS = list(range(2020, 2025))  # 2020-2024 inclusive
AIME_VALIDATION_YEARS = list(range(1983, 2020))  # 1983-2019 inclusive


def extract_boxed_answer(text: str) -> str:
    """Extract answer from LaTeX \\boxed{} notation or last sentence."""
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, text, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()

    sentence_end_pattern = r"(?<!\d)[.!?]\s+"
    sentences = re.split(sentence_end_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else ""


def get_validation_cache_path(dataset_type: str, num_problems: int) -> Path:
    """Generate deterministic cache path for validation data."""
    cache_filename = f"validation_{dataset_type}_seed{VALIDATION_SEED}_size{num_problems}.json"
    return VALIDATION_CACHE_DIR / cache_filename


def get_test_cache_path(dataset_type: str) -> Path:
    """Generate deterministic cache path for test data."""
    if dataset_type == "musique":
        cache_filename = f"musique_test_seed{TEST_SEED}_size500.json"
    else:
        cache_filename = f"test_{dataset_type}.json"
    return TEST_CACHE_DIR / cache_filename


def save_validation_cache(dataset_type: str, num_problems: int, examples: List[Dict]) -> None:
    """Save validation examples to cache file."""
    try:
        cache_path = get_validation_cache_path(dataset_type, num_problems)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        cache_data = {
            "dataset_type": dataset_type,
            "validation_seed": VALIDATION_SEED,
            "num_problems": num_problems,
            "total_examples": len(examples),
            "examples": examples,
            "metadata": {
                "created_at": datetime.now().isoformat()
            }
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Saved validation cache: {cache_path} ({len(examples)} examples)")
        
    except Exception as e:
        print(f"Warning: Could not save validation cache: {e}")


def save_test_cache(dataset_type: str, examples: List[Dict]) -> None:
    """Save test examples to cache file."""
    try:
        cache_path = get_test_cache_path(dataset_type)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        cache_data = {
            "dataset_type": dataset_type,
            "total_examples": len(examples),
            "examples": examples,
            "metadata": {
                "created_at": datetime.now().isoformat()
            }
        }
        
        # Add dataset-specific metadata
        if dataset_type == "musique":
            cache_data["test_seed"] = TEST_SEED
            cache_data["num_problems"] = len(examples)
        elif dataset_type == "aime":
            cache_data["metadata"]["test_years"] = AIME_TEST_YEARS
            cache_data["metadata"]["validation_years_max"] = max(AIME_VALIDATION_YEARS)
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Saved test cache: {cache_path} ({len(examples)} examples)")
        
    except Exception as e:
        print(f"Warning: Could not save test cache: {e}")


def load_validation_cache(dataset_type: str, num_problems: int) -> Optional[List[Dict]]:
    """Load validation examples from cache if available."""
    try:
        cache_path = get_validation_cache_path(dataset_type, num_problems)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Validate cache metadata
        if (cache_data.get("dataset_type") == dataset_type and
            cache_data.get("validation_seed") == VALIDATION_SEED and
            cache_data.get("num_problems") == num_problems):
            
            examples = cache_data.get("examples", [])
            print(f"Loaded validation cache: {cache_path} ({len(examples)} examples)")
            return examples
        else:
            print(f"Cache metadata mismatch, regenerating validation set")
            return None
            
    except Exception as e:
        print(f"Warning: Could not load validation cache: {e}")
        return None


def load_test_cache(dataset_type: str) -> Optional[List[Dict]]:
    """Load test examples from cache if available."""
    try:
        cache_path = get_test_cache_path(dataset_type)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Validate cache metadata based on dataset type
        if dataset_type == "musique":
            if (cache_data.get("dataset_type") == "musique" and
                cache_data.get("test_seed") == TEST_SEED and
                cache_data.get("num_problems") == 500):
                
                examples = cache_data.get("examples", [])
                print(f"Loaded MuSiQue test cache: {cache_path} ({len(examples)} examples)")
                return examples
            else:
                print(f"Cache metadata mismatch, regenerating MuSiQue test set")
                return None
        else:
            if cache_data.get("dataset_type") == dataset_type:
                examples = cache_data.get("examples", [])
                print(f"Loaded test cache: {cache_path} ({len(examples)} examples)")
                
                # Print metadata for AIME
                if dataset_type == "aime" and "metadata" in cache_data:
                    meta = cache_data["metadata"]
                    if "test_years" in meta:
                        print(f"  Test years: {meta['test_years']}")
                
                return examples
            else:
                print(f"Cache dataset type mismatch, regenerating test set")
                return None
            
    except Exception as e:
        print(f"Warning: Could not load test cache: {e}")
        return None


def _login_huggingface():
    """Attempt to login to HuggingFace if token is available."""
    try:
        with open('/workspace/hf_key', 'r') as f:
            hf_token = f.read().strip()
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")
        print("Attempting to access dataset without authentication...")


def _shuffle_choices_and_format_deterministic(row, base_seed=VALIDATION_SEED):
    """
    Shuffle answer choices deterministically using fixed seed + question hash.
    For GPQA multiple choice questions.
    """
    # Use hashlib for consistent hashing across program runs
    question_bytes = row['Question'].encode('utf-8')
    question_hash = int(hashlib.md5(question_bytes).hexdigest()[:8], 16)
    
    # Use base seed + question hash for reproducible but unique per-question shuffling
    question_seed = base_seed + (question_hash % 1000)
    random.seed(question_seed)
    
    # Create list of all choices
    choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], 
              row['Incorrect Answer 3'], row['Correct Answer']]
    
    # Shuffle the choices deterministically
    random.shuffle(choices)
    
    # Find which position the correct answer ended up in
    correct_position = choices.index(row['Correct Answer'])
    
    # Convert to letter (A=0, B=1, C=2, D=3)
    correct_letter = ['A', 'B', 'C', 'D'][correct_position]
    
    # Format question with shuffled options
    question = row["Question"]
    options = f"\n\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
    formatted_question = question + options
    
    return formatted_question, correct_letter


def format_musique_problem(paragraphs: List[Dict], question: str) -> str:
    """
    Format MuSiQue problem with all paragraphs and question.
    
    Args:
        paragraphs: List of paragraph dictionaries with title and paragraph_text
        question: The question to answer
    
    Returns:
        Formatted problem string
    """
    formatted_parts = ["Read the following paragraphs and then answer the question:\n"]
    
    for paragraph in paragraphs:
        title = paragraph.get('title', 'Untitled')
        text = paragraph.get('paragraph_text', '')
        
        formatted_parts.append("----")
        formatted_parts.append(f"Title: {title}")
        formatted_parts.append("----")
        formatted_parts.append(text)
        formatted_parts.append("")  # Add blank line for double newline
    
    formatted_parts.append(f"Question: {question}")
    
    return "\n".join(formatted_parts)


# MATH Dataset Functions (unchanged)
def get_math_validation_examples(num_problems: int) -> List[Dict[str, str]]:
    """Load MATH validation problems with deterministic caching."""
    # Try to load from cache first
    cached_examples = load_validation_cache("math", num_problems)
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating new MATH validation set with seed={VALIDATION_SEED}, size={num_problems}")
    
    try:
        dataset = load_dataset('hiyouga/math12k', split='train')
        all_examples = [{"problem": item["problem"], "answer": item["answer"]} for item in dataset]
        
        # Use fixed seed for deterministic sampling
        random.seed(VALIDATION_SEED)
        if len(all_examples) > num_problems:
            selected_indices = sorted(random.sample(range(len(all_examples)), num_problems))
            examples = [all_examples[i] for i in selected_indices]
        else:
            examples = all_examples
        
        # Save to cache for future runs
        save_validation_cache("math", num_problems, examples)
        
        return examples
        
    except Exception as e:
        print(f"Failed to load MATH dataset: {e}")
        return []


def get_math_test_examples_alternate() -> List[Dict[str, str]]:
    """
    Load MATH test problems from local JSONL file with caching.
    
    This is an alternative implementation that loads from:
        local_datasets/math_test.jsonl
        This is the MATH level 5 test set from the AFlow paper
    
    Returns:
        List of dictionaries with 'problem' and 'answer' keys
    """
    # Try to load from cache first
    cached_examples = load_test_cache("math")
    if cached_examples is not None:
        return cached_examples
    
    # Path to the local JSONL file
    jsonl_path = Path("local_datasets/math_test.jsonl")
    
    print(f"Loading MATH test examples from: {jsonl_path}")
    
    try:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"MATH test file not found at: {jsonl_path}")
        
        examples = []
        
        # Read and parse the JSONL file
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Extract problem text
                    problem = data.get('problem', '')
                    if not problem:
                        print(f"Warning: Line {line_num} missing 'problem' field, skipping")
                        continue
                    
                    # Extract solution and then extract answer from it
                    solution = data.get('solution', '')
                    if not solution:
                        print(f"Warning: Line {line_num} missing 'solution' field, skipping")
                        continue
                    
                    # Use the extract_boxed_answer function to get the answer
                    answer = extract_boxed_answer(solution)
                    
                    # Add to examples list
                    examples.append({
                        "problem": problem,
                        "answer": answer
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
        
        if not examples:
            raise ValueError("No valid examples found in the JSONL file")
        
        print(f"Successfully loaded {len(examples)} MATH test examples")
        
        # Save to cache for future runs
        save_test_cache("math", examples)
        
        return examples
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Falling back to HuggingFace dataset...")
        # Fall back to the original implementation
        return get_math_test_examples_original()
        
    except Exception as e:
        print(f"Failed to load MATH test dataset from local file: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to HuggingFace dataset...")
        # Fall back to the original implementation
        return get_math_test_examples_original()


def get_math_test_examples_original() -> List[Dict[str, str]]:
    """
    Original implementation that loads from HuggingFace.
    Used as fallback if local file is not available.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')
        examples = [{"problem": item["problem"], "answer": item["answer"]} for item in dataset]
        return examples
    except Exception as e:
        print(f"Failed to load MATH-500 dataset from HuggingFace: {e}")
        return []


# Updated main function to use the alternate implementation
def get_math_test_examples() -> List[Dict[str, str]]:
    """
    Load MATH test problems with caching.
    First tries to load from local JSONL file, falls back to HuggingFace if needed.
    """
    return get_math_test_examples_alternate()

'''
def get_math_test_examples() -> List[Dict[str, str]]:
    """Load MATH test problems with caching."""
    # Try to load from cache first
    cached_examples = load_test_cache("math")
    if cached_examples is not None:
        return cached_examples
    
    try:
        dataset = load_dataset('HuggingFaceH4/MATH-500', split='test')
        examples = [{"problem": item["problem"], "answer": item["answer"]} for item in dataset]
        
        # Save to cache for future runs
        save_test_cache("math", examples)
        
        return examples
    except Exception as e:
        print(f"Failed to load MATH-500 dataset: {e}")
        return []
'''

# GPQA Dataset Functions (unchanged)
def get_gpqa_validation_examples(num_problems: int) -> List[Dict[str, str]]:
    """Load GPQA validation problems with deterministic caching."""
    # Try to load from cache first
    cached_examples = load_validation_cache("gpqa", num_problems)
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating new GPQA validation set with seed={VALIDATION_SEED}, size={num_problems}")
    
    try:
        _login_huggingface()
        
        import pandas as pd
        
        # Load both datasets
        print("Loading GPQA Main...")
        gpqa_main_df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_main.csv")
        print(f"Loaded {len(gpqa_main_df)} questions from GPQA Main")
        
        print("Loading GPQA Diamond...")
        gpqa_diamond_df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")
        print(f"Loaded {len(gpqa_diamond_df)} questions from GPQA Diamond")
        
        # Deterministic validation set selection (Main - Diamond)
        main_questions = set(gpqa_main_df['Question'].tolist())
        diamond_questions = set(gpqa_diamond_df['Question'].tolist())
        
        validation_questions = main_questions - diamond_questions
        validation_df = gpqa_main_df[gpqa_main_df['Question'].isin(validation_questions)]
        
        # Sort by question text for deterministic ordering
        validation_df = validation_df.sort_values('Question').reset_index(drop=True)
        
        print(f"GPQA validation pool size: {len(validation_df)} questions")
        
        # Format all examples
        all_examples = []
        for _, row in validation_df.iterrows():
            formatted_question, correct_letter = _shuffle_choices_and_format_deterministic(row)
            all_examples.append({"problem": formatted_question, "answer": correct_letter})
        
        # Sample if needed
        random.seed(VALIDATION_SEED)
        if len(all_examples) > num_problems:
            selected_indices = sorted(random.sample(range(len(all_examples)), num_problems))
            examples = [all_examples[i] for i in selected_indices]
        else:
            examples = all_examples
        
        print(f"Selected {len(examples)} GPQA validation examples")
        
        # Save to cache for future runs
        save_validation_cache("gpqa", num_problems, examples)
        
        return examples
               
    except Exception as e:
        print(f"Failed to load GPQA dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_gpqa_test_examples() -> List[Dict[str, str]]:
    """Load GPQA test problems with caching."""
    # Try to load from cache first
    cached_examples = load_test_cache("gpqa")
    if cached_examples is not None:
        return cached_examples
    
    try:
        _login_huggingface()
        
        import pandas as pd
        
        print("Loading GPQA Diamond for test...")
        gpqa_diamond_df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")
        print(f"Loaded {len(gpqa_diamond_df)} questions from GPQA Diamond for test")
        
        # Format all examples
        examples = []
        for _, row in gpqa_diamond_df.iterrows():
            formatted_question, correct_letter = _shuffle_choices_and_format_deterministic(row)
            examples.append({"problem": formatted_question, "answer": correct_letter})
        
        # Save to cache for future runs
        save_test_cache("gpqa", examples)
        
        return examples
        
    except Exception as e:
        print(f"Failed to load GPQA Diamond dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


# AIME Dataset Functions (MODIFIED)
def get_aime_validation_examples(num_problems: int) -> List[Dict[str, str]]:
    """Load AIME validation problems from years 1983-2019 with deterministic caching."""
    # Try to load from cache first
    cached_examples = load_validation_cache("aime", num_problems)
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating new AIME validation set with seed={VALIDATION_SEED}, size={num_problems}")
    print(f"Using problems from years: {min(AIME_VALIDATION_YEARS)}-{max(AIME_VALIDATION_YEARS)}")
    
    try:
        dataset = load_dataset('gneubig/aime-1983-2024', split='train')
        
        # Filter to only validation years (pre-2020)
        validation_examples = []
        for item in dataset:
            # Extract year from the problem - the dataset has a 'Year' field
            year = item.get('Year', None)
            
            if year is None:
                # If no Year field, skip this problem
                continue
            
            if year in AIME_VALIDATION_YEARS:
                validation_examples.append({
                    "problem": item["Question"],
                    "answer": str(item["Answer"]),
                    "year": year  # Store year for reference
                })
        
        if not validation_examples:
            raise ValueError(f"No AIME problems found for years {min(AIME_VALIDATION_YEARS)}-{max(AIME_VALIDATION_YEARS)}")
        
        print(f"Found {len(validation_examples)} problems from validation years")
        
        # Use fixed seed for deterministic sampling
        random.seed(VALIDATION_SEED)
        if len(validation_examples) > num_problems:
            selected_indices = sorted(random.sample(range(len(validation_examples)), num_problems))
            examples = [validation_examples[i] for i in selected_indices]
        else:
            examples = validation_examples
        
        # Remove year from final output but keep problem and answer
        examples_for_output = [
            {"problem": ex["problem"], "answer": ex["answer"]} 
            for ex in examples
        ]
        
        print(f"Selected {len(examples)} AIME validation examples")
        
        # Save to cache for future runs
        save_validation_cache("aime", num_problems, examples_for_output)
        
        return examples_for_output
        
    except Exception as e:
        print(f"Failed to load AIME dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_aime_test_examples() -> List[Dict[str, str]]:
    """Load AIME test problems from years 2020-2024 with caching."""
    # Try to load from cache first
    cached_examples = load_test_cache("aime")
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating new AIME test set from years: {AIME_TEST_YEARS}")
    
    try:
        dataset = load_dataset('gneubig/aime-1983-2024', split='train')
        
        # Filter to only test years (2020-2024)
        test_examples = []
        for item in dataset:
            # Extract year from the problem - the dataset has a 'Year' field
            year = item.get('Year', None)
            
            if year is None:
                # If no Year field, skip this problem
                continue
            
            if year in AIME_TEST_YEARS:
                test_examples.append({
                    "problem": item["Question"],
                    "answer": str(item["Answer"]),
                    "year": year  # Preserve year metadata
                })
        
        if not test_examples:
            raise ValueError(f"No AIME problems found for test years {AIME_TEST_YEARS}. "
                           f"Dataset may not contain these years.")
        
        print(f"Found {len(test_examples)} problems from test years")
        
        # Group by year for statistics
        from collections import defaultdict
        year_counts = defaultdict(int)
        for ex in test_examples:
            year_counts[ex["year"]] += 1
        
        print("Problems per year:")
        for year in sorted(year_counts.keys()):
            print(f"  {year}: {year_counts[year]} problems")
        
        # Shuffle test examples randomly (but deterministically for reproducibility)
        random.seed(42)  # Use a fixed seed for test set shuffling
        random.shuffle(test_examples)
        
        # Format for output (keep year in metadata but not in main fields)
        examples_for_output = []
        for ex in test_examples:
            examples_for_output.append({
                "problem": ex["problem"],
                "answer": ex["answer"],
                "metadata": {"year": ex["year"]}  # Store year in metadata
            })
        
        print(f"Total AIME test problems: {len(examples_for_output)}")
        
        # Save to cache for future runs
        save_test_cache("aime", examples_for_output)
        
        return examples_for_output
        
    except Exception as e:
        print(f"Failed to load AIME test dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


# MuSiQue Dataset Functions (unchanged except for test caching)
def get_musique_validation_examples(num_problems: int) -> List[Dict[str, str]]:
    """Load MuSiQue validation problems from train split with deterministic caching."""
    # Try to load from cache first
    cached_examples = load_validation_cache("musique", num_problems)
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating new MuSiQue validation set with seed={VALIDATION_SEED}, size={num_problems}")
    
    try:
        # Load train split for validation
        dataset = load_dataset('dgslibisey/MuSiQue', split='train')
        
        # Format all examples
        all_examples = []
        for item in dataset:
            formatted_problem = format_musique_problem(item['paragraphs'], item['question'])
            all_examples.append({
                "problem": formatted_problem,
                "answer": item['answer']
            })
        
        print(f"Total MuSiQue train examples: {len(all_examples)}")
        
        # Use fixed seed for deterministic sampling
        random.seed(VALIDATION_SEED)
        if len(all_examples) > num_problems:
            selected_indices = sorted(random.sample(range(len(all_examples)), num_problems))
            examples = [all_examples[i] for i in selected_indices]
        else:
            examples = all_examples
        
        print(f"Selected {len(examples)} MuSiQue validation examples")
        
        # Save to cache for future runs
        save_validation_cache("musique", num_problems, examples)
        
        return examples
        
    except Exception as e:
        print(f"Failed to load MuSiQue dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_musique_test_examples() -> List[Dict[str, str]]:
    """Load fixed 500 MuSiQue test problems from validation split with caching."""
    # Try to load from cache first
    cached_examples = load_test_cache("musique")
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating MuSiQue test set with seed={TEST_SEED}, size=500")
    
    try:
        # Load validation split for test
        dataset = load_dataset('dgslibisey/MuSiQue', split='validation')
        
        # Format all examples
        all_examples = []
        for item in dataset:
            formatted_problem = format_musique_problem(item['paragraphs'], item['question'])
            all_examples.append({
                "problem": formatted_problem,
                "answer": item['answer']
            })
        
        print(f"Total MuSiQue validation examples: {len(all_examples)}")
        
        # Use fixed seed for deterministic sampling of 500 problems
        random.seed(TEST_SEED)
        if len(all_examples) > 500:
            selected_indices = sorted(random.sample(range(len(all_examples)), 500))
            examples = [all_examples[i] for i in selected_indices]
        else:
            examples = all_examples
        
        print(f"Selected {len(examples)} MuSiQue test examples")
        
        # Save to cache for future runs
        save_test_cache("musique", examples)
        
        return examples
        
    except Exception as e:
        print(f"Failed to load MuSiQue test dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def format_mmlupro_question(question_data: Dict, base_seed: int = VALIDATION_SEED) -> Tuple[str, str]:
    """
    Format MMLU-Pro question, handling "N/A" options properly.
    
    Args:
        question_data: Dictionary with 'question', 'options', 'answer' (letter)
        base_seed: Base seed for shuffling
    
    Returns:
        Tuple of (formatted_question, correct_answer_label)
    """
    question = question_data['question']
    options = question_data['options']  # List of option texts, some might be "N/A"
    answer_letter = question_data['answer']  # Letter like "A", "B", etc.
    
    # Filter out "N/A" options and keep track of valid options
    valid_options = []
    original_indices = []
    
    for i, option in enumerate(options):
        if option != "N/A" and option.strip():  # Skip "N/A" and empty options
            valid_options.append(option)
            original_indices.append(i)
    
    # Skip if we have less than 2 valid options (not a real MCQ)
    if len(valid_options) < 2:
        return None, None
    
    # Convert answer letter to index to get the correct option text
    if answer_letter and isinstance(answer_letter, str) and len(answer_letter) == 1:
        answer_index = ord(answer_letter.upper()) - ord('A')
        if 0 <= answer_index < len(options):
            correct_answer_text = options[answer_index]
            # Make sure the correct answer is not "N/A"
            if correct_answer_text == "N/A":
                print(f"Warning: Correct answer points to 'N/A' option")
                return None, None
        else:
            print(f"Warning: Invalid answer letter '{answer_letter}' for {len(options)} options")
            return None, None
    else:
        print(f"Warning: Unexpected answer format: {answer_letter}")
        return None, None
    
    # Create labels based on number of valid options
    num_valid = len(valid_options)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][:num_valid]
    
    # Deterministic shuffle based on question hash
    question_bytes = question.encode('utf-8')
    question_hash = int(hashlib.md5(question_bytes).hexdigest()[:8], 16)
    question_seed = base_seed + (question_hash % 1000)
    random.seed(question_seed)
    
    # Shuffle the valid options
    shuffled_options = valid_options.copy()
    random.shuffle(shuffled_options)
    
    # Find where the correct answer text ended up after shuffle
    correct_label = None
    for i, option_text in enumerate(shuffled_options):
        if option_text == correct_answer_text:
            correct_label = labels[i]
            break
    
    if correct_label is None:
        print(f"Warning: Could not find correct answer in shuffled options")
        return None, None
    
    # Format question with all valid options
    formatted = question
    if not question.endswith('?') and not question.endswith('.'):
        formatted += '.'
    
    formatted += "\n\n"
    for i, option_text in enumerate(shuffled_options):
        formatted += f"({labels[i]}) {option_text}\n"
    
    return formatted.strip(), correct_label


def load_and_split_mmlupro_test_set() -> Tuple[List[Dict], List[Dict]]:
    """
    Load MMLU-Pro test set and split it deterministically into validation and test.
    
    Returns:
        Tuple of (validation_items, test_items)
    """
    # Load the full test set
    dataset = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    
    # Debug first item structure
    if len(dataset) > 0:
        first_item = dataset[0]
        print(f"MMLU-Pro dataset structure:")
        print(f"  Keys: {first_item.keys()}")
        print(f"  Question: {first_item['question'][:100]}...")
        print(f"  Options: {first_item.get('options', [])}")
        print(f"  Answer: '{first_item.get('answer')}' (type: {type(first_item.get('answer'))})")
    
    # Filter to questions with at least 2 valid (non-"N/A") options
    filtered_items = []
    for item in dataset:
        options = item.get('options', [])
        valid_options = [opt for opt in options if opt != "N/A" and opt.strip()]
        
        # Need at least 2 valid options and a valid answer
        if len(valid_options) >= 2 and item.get('answer'):
            # Check that the answer doesn't point to "N/A"
            answer_letter = item['answer']
            if answer_letter and isinstance(answer_letter, str) and len(answer_letter) == 1:
                answer_index = ord(answer_letter.upper()) - ord('A')
                if 0 <= answer_index < len(options) and options[answer_index] != "N/A":
                    filtered_items.append(item)
    
    print(f"Total MMLU-Pro test questions with valid options and answers: {len(filtered_items)}")
    
    # Create deterministic split using hash of item content
    validation_items = []
    test_items = []
    
    for item in filtered_items:
        # Create stable hash for this item
        item_str = f"{item['question']}_{item.get('answer', '')}"
        item_hash = int(hashlib.md5(item_str.encode()).hexdigest()[:8], 16)
        
        # Use modulo to deterministically assign to validation or test
        # This gives us roughly 50/50 split
        if item_hash % 2 == 0:
            validation_items.append(item)
        else:
            test_items.append(item)
    
    print(f"Split into {len(validation_items)} validation and {len(test_items)} test items")
    
    return validation_items, test_items


# MMLU-Pro Dataset Functions
def get_mmlupro_validation_examples(num_problems: int) -> List[Dict[str, str]]:
    """Load MMLU-Pro validation problems from test set split with deterministic caching."""
    # Try to load from cache first
    cached_examples = load_validation_cache("mmlupro", num_problems)
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating new MMLU-Pro validation set with seed={VALIDATION_SEED}, size={num_problems}")
    
    try:
        # Get validation split from test set
        validation_items, _ = load_and_split_mmlupro_test_set()
        
        # Format all examples
        all_examples = []
        for item in validation_items:
            question_data = {
                'question': item['question'],
                'options': item['options'],
                'answer': item['answer']
            }
            
            formatted_question, correct_label = format_mmlupro_question(
                question_data, VALIDATION_SEED
            )
            
            all_examples.append({
                "problem": formatted_question,
                "answer": correct_label
            })
        
        print(f"Total MMLU-Pro validation examples available: {len(all_examples)}")
        
        # Use fixed seed for deterministic sampling
        random.seed(VALIDATION_SEED)
        if len(all_examples) > num_problems:
            selected_indices = sorted(random.sample(range(len(all_examples)), num_problems))
            examples = [all_examples[i] for i in selected_indices]
        else:
            examples = all_examples
        
        print(f"Selected {len(examples)} MMLU-Pro validation examples")
        
        # Save to cache for future runs
        save_validation_cache("mmlupro", num_problems, examples)
        
        return examples
        
    except Exception as e:
        print(f"Failed to load MMLU-Pro dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_mmlupro_test_examples() -> List[Dict[str, str]]:
    """Load fixed 500 MMLU-Pro test problems from test set split with caching."""
    # Try to load from cache first
    cached_examples = load_test_cache("mmlupro")
    if cached_examples is not None:
        return cached_examples
    
    print(f"Generating MMLU-Pro test set with seed={TEST_SEED}, size=500")
    
    try:
        # Get test split from test set
        _, test_items = load_and_split_mmlupro_test_set()
        
        # Format all examples
        all_examples = []
        for item in test_items:
            question_data = {
                'question': item['question'],
                'options': item['options'],
                'answer': item['answer']
            }
            
            formatted_question, correct_label = format_mmlupro_question(
                question_data, TEST_SEED
            )
            
            all_examples.append({
                "problem": formatted_question,
                "answer": correct_label
            })
        
        print(f"Total MMLU-Pro test examples available: {len(all_examples)}")
        
        # Use fixed seed for deterministic sampling of 500 problems
        random.seed(TEST_SEED)
        if len(all_examples) > 500:
            selected_indices = sorted(random.sample(range(len(all_examples)), 500))
            examples = [all_examples[i] for i in selected_indices]
        else:
            examples = all_examples
        
        print(f"Selected {len(examples)} MMLU-Pro test examples")
        
        # Save to cache for future runs
        save_test_cache("mmlupro", examples)
        
        return examples
        
    except Exception as e:
        print(f"Failed to load MMLU-Pro test dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


# Unified interface functions
def get_validation_examples(dataset_type: str, num_problems: int) -> List[Dict[str, str]]:
    """Get validation examples for specified dataset type."""
    if dataset_type == "math":
        return get_math_validation_examples(num_problems)
    elif dataset_type == "gpqa":
        return get_gpqa_validation_examples(num_problems)
    elif dataset_type == "aime":
        return get_aime_validation_examples(num_problems)
    elif dataset_type == "musique":
        return get_musique_validation_examples(num_problems)
    elif dataset_type == "mmlupro":
        return get_mmlupro_validation_examples(num_problems)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_test_examples(dataset_type: str) -> List[Dict[str, str]]:
    """Get test examples for specified dataset type."""
    if dataset_type == "math":
        return get_math_test_examples()
    elif dataset_type == "gpqa":
        return get_gpqa_test_examples()
    elif dataset_type == "aime":
        return get_aime_test_examples()
    elif dataset_type == "musique":
        return get_musique_test_examples()
    elif dataset_type == "mmlupro":
        return get_mmlupro_test_examples()
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def clear_validation_cache(dataset_type: str = None):
    """Clear validation cache files."""
    if not VALIDATION_CACHE_DIR.exists():
        return
    
    try:
        if dataset_type:
            # Clear specific dataset cache
            pattern = f"validation_{dataset_type}_*"
            import glob
            cache_files = glob.glob(str(VALIDATION_CACHE_DIR / pattern))
        else:
            # Clear all cache files
            cache_files = list(VALIDATION_CACHE_DIR.glob("*.json"))
        
        for cache_file in cache_files:
            Path(cache_file).unlink()
            print(f"Removed cache file: {cache_file}")
        
        # Remove cache directory if empty
        if not any(VALIDATION_CACHE_DIR.iterdir()):
            VALIDATION_CACHE_DIR.rmdir()
            print(f"Removed empty cache directory: {VALIDATION_CACHE_DIR}")
            
    except Exception as e:
        print(f"Error clearing cache: {e}")


def clear_test_cache(dataset_type: str = None):
    """Clear test cache files."""
    if not TEST_CACHE_DIR.exists():
        return
    
    try:
        if dataset_type:
            # Clear specific dataset cache
            cache_file = get_test_cache_path(dataset_type)
            if cache_file.exists():
                cache_file.unlink()
                print(f"Removed test cache file: {cache_file}")
        else:
            # Clear all test cache files
            cache_files = list(TEST_CACHE_DIR.glob("*.json"))
            for cache_file in cache_files:
                cache_file.unlink()
                print(f"Removed test cache file: {cache_file}")
        
        # Remove cache directory if empty
        if TEST_CACHE_DIR.exists() and not any(TEST_CACHE_DIR.iterdir()):
            TEST_CACHE_DIR.rmdir()
            print(f"Removed empty cache directory: {TEST_CACHE_DIR}")
            
    except Exception as e:
        print(f"Error clearing test cache: {e}")
