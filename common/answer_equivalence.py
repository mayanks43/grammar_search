"""
Unified answer equivalence checking for mathematical reasoning and text-based QA.
Consolidates duplicated logic from across the codebase.
Enhanced with simple cache to reduce LLM judge calls.
Now includes the centralized judge_answer function with dataset-specific routing.
Updated with token tracking, cost monitoring, and plaintext answer judging for MuSiQue.
Enhanced with concrete examples in judge prompts for better accuracy.
Modified to use plaintext true/false responses instead of JSON format.
Updated to store full question text in cache for debugging.
"""

import json
import backoff
import openai
import time
import random
from common.config import JUDGE_MODEL
from common.llm_interface import _judge_answer_with_llm
from common.debug_logger import debug_logger
from common.token_tracker import token_tracker
from common.azure_utils import azure_retry_backoff

import hashlib
from typing import Optional, Dict, Tuple

# Modified cache structure to include question text
_equivalence_cache: Dict[Tuple[str, str, str, Optional[str]], Tuple[bool, str]] = {}


def _hash_question(question: str) -> str:
    """
    Hash the question using SHA-256.
    This reduces memory usage for large questions while maintaining near-zero collision probability.
    """
    if not question:
        return ""
    
    # SHA-256 gives us 2^256 possible values - collision probability is negligible
    hash_object = hashlib.sha256(question.encode('utf-8'))
    return hash_object.hexdigest()


def _create_cache_key(answer1: str, answer2: str, question: str = "", dataset_type: Optional[str] = None) -> Tuple[str, str, str, Optional[str]]:
    """
    Create cache key with normalized answers and hashed question.
    Ensures symmetry by ordering answers consistently.
    
    Args:
        answer1: The ground truth answer
        answer2: The prediction answer to check
    """
    # Normalize answers for comparison
    norm_answer1 = answer1.strip().lower()
    norm_answer2 = answer2.strip().lower()
    
    # Hash the question to save memory
    question_hash = _hash_question(question)
    
    # Sort answers to ensure consistent ordering - this handles symmetry
    # We want cache to work for both (A, B) and (B, A) since equivalence is symmetric
    if norm_answer1 <= norm_answer2:
        return (norm_answer1, norm_answer2, question_hash, dataset_type)
    else:
        return (norm_answer2, norm_answer1, question_hash, dataset_type)


def judge_math_answer(ground_truth: str, final_answer: str, question: str) -> bool:
    """Use a powerful LLM to judge mathematical equivalence."""
    system_message = (
        "You are an expert mathematician and a fair judge. Your task is to determine if the proposed answer is mathematically equivalent to the ground truth answer for the given problem. "
        "Account for different but equivalent formats (e.g., 0.5 vs 1/2). "
        "Respond with 'true' if the answers match and 'false' if they don't.\n\n"
        "Examples:\n\n"
        "Example 1 - Equivalent algebraic expressions:\n"
        "Question: Simplify the expression 2x + 4\n"
        "Correct Answer: 2(x + 2)\n"
        "Proposed Answer: 2x + 4\n"
        "Response: true\n\n"
        "Example 2 - Different numerical forms:\n"
        "Question: What is the square root of 2?\n"
        "Correct Answer: sqrt(2)\n"
        "Proposed Answer: 1.41421356\n"
        "Response: true\n\n"
        "Example 3 - Correct answer embedded in text:\n"
        "Question: What is 15 × 28?\n"
        "Correct Answer: 420\n"
        "Proposed Answer: After multiplying 15 by 28, I get 420 which is the final answer.\n"
        "Response: true\n\n"
        "Example 4 - Completely unrelated answer:\n"
        "Question: Find the derivative of x^2\n"
        "Correct Answer: 2x\n"
        "Proposed Answer: The weather is nice today.\n"
        "Response: false\n\n"
        "Example 5 - Wrong answer:\n"
        "Question: Solve for x: 2x + 6 = 14\n"
        "Correct Answer: x = 4\n"
        "Proposed Answer: x = 7\n"
        "Response: false"
    )
    return _judge_answer_with_llm(
        ground_truth,
        final_answer,
        question,
        JUDGE_MODEL,
        system_message,
        "Math Judge System"
    )


def judge_mcq_answer(ground_truth: str, final_answer: str, question: str) -> bool:
    """Judge multiple choice answers using LLM only."""
    system_message = (
        "You are an expert judge for multiple choice questions. Your task is to determine if the proposed answer matches the correct answer. "
        "The answers should be A, B, C, or D. Account for different formatting. For example, if the correct answer is A, accept: "
        "'A', '(A)', 'A.', 'A)', 'Option A', 'Choice A', 'The answer is A', "
        "'(A) First option text', 'A - First option text', or even just the full text of the correct option, and other similar variations. "
        "Apply the same logic for B, C, and D when they are the correct answers. "
        "Be flexible in recognizing equivalent answers regardless of formatting or verbosity. "
        "Respond with 'true' if the answers match and 'false' if they don't.\n\n"
        "Examples:\n\n"
        "Example 1 - Various correct formats:\n"
        "Question: What is the capital of France? A) London B) Paris C) Berlin D) Madrid\n"
        "Correct Answer: B\n"
        "Proposed Answer: (B) Paris\n"
        "Response: true\n\n"
        "Example 2 - Full text answer:\n"
        "Question: Which planet is closest to the Sun? A) Venus B) Mercury C) Earth D) Mars\n"
        "Correct Answer: B\n"
        "Proposed Answer: Mercury is the closest planet to the Sun\n"
        "Response: true\n\n"
        "Example 3 - Answer embedded in explanation:\n"
        "Question: What is 2+2? A) 3 B) 4 C) 5 D) 6\n"
        "Correct Answer: B\n"
        "Proposed Answer: The answer is clearly option B, which is 4.\n"
        "Response: true\n\n"
        "Example 4 - Wrong answer:\n"
        "Question: What color is the sky? A) Red B) Blue C) Green D) Yellow\n"
        "Correct Answer: B\n"
        "Proposed Answer: C\n"
        "Response: false\n\n"
        "Example 5 - Completely unrelated text:\n"
        "Question: What is the largest ocean? A) Atlantic B) Pacific C) Indian D) Arctic\n"
        "Correct Answer: B\n"
        "Proposed Answer: I like pizza.\n"
        "Response: false"
    )
    return _judge_answer_with_llm(
        ground_truth,
        final_answer,
        question,
        JUDGE_MODEL,
        system_message,
        "MCQ Judge System"
    )


def judge_mmlupro_answer(ground_truth: str, final_answer: str, question: str) -> bool:
    """Judge MMLU-Pro answers with variable number of choices."""
    system_message = (
        "You are an expert judge for multiple choice questions. "
        "The questions may have anywhere from 2 to 10 options (labeled with letters A, B, C, etc.). "
        "Your task is to determine if the proposed answer matches the correct answer. "
        "Account for different formatting. For example, if the correct answer is C, accept: "
        "'C', '(C)', 'C.', 'C)', 'Option C', 'Choice C', 'The answer is C', "
        "'(C) [actual option text]', 'C - [actual option text]', or even just the full text of the correct option, "
        "and other similar variations. "
        "Be flexible in recognizing equivalent answers regardless of formatting or verbosity. "
        "Respond with 'true' if the answers match and 'false' if they don't.\n\n"
        "Examples:\n\n"
        "Example 1 - Various correct formats:\n"
        "Question: What is the capital of France? A) London B) Paris C) Berlin D) Rome\n"
        "Correct Answer: B\n"
        "Proposed Answer: (B) Paris\n"
        "Response: true\n\n"
        "Example 2 - Full text answer:\n"
        "Question: Which element has atomic number 79? A) Silver B) Gold C) Iron\n"
        "Correct Answer: B\n"
        "Proposed Answer: Gold is the element with atomic number 79\n"
        "Response: true\n\n"
        "Example 3 - Answer embedded in explanation:\n"
        "Question: What is 15 × 28? A) 320 B) 420 C) 440 D) 460 E) 480\n"
        "Correct Answer: B\n"
        "Proposed Answer: After multiplying 15 by 28, I get option B, which is 420.\n"
        "Response: true\n\n"
        "Example 4 - Wrong answer:\n"
        "Question: What is the square root of 144? A) 10 B) 11 C) 12 D) 13\n"
        "Correct Answer: C\n"
        "Proposed Answer: A\n"
        "Response: false\n\n"
        "Example 5 - Completely unrelated text:\n"
        "Question: Which year did World War II end? A) 1943 B) 1944 C) 1945\n"
        "Correct Answer: C\n"
        "Proposed Answer: I enjoy reading history books.\n"
        "Response: false"
    )
    return _judge_answer_with_llm(
        ground_truth,
        final_answer,
        question,
        JUDGE_MODEL,
        system_message,
        "MMLU-Pro Judge System"
    )


def judge_plaintext_answer(ground_truth: str, final_answer: str, question: str) -> bool:
    """
    Judge plaintext answers for factual questions using LLM.
    Used for MuSiQue and other short factual text answers.
    """
    system_message = (
        "You are an expert judge for factual question-answering. Your task is to determine if the proposed answer is correct based on the ground truth answer. "
        "The proposed answer should convey the same factual information as the ground truth, but may differ in:\n"
        "- Capitalization (e.g., 'paris' vs 'Paris')\n"
        "- Articles (e.g., 'the president' vs 'president', 'a dog' vs 'dog')\n"
        "- Minor punctuation differences\n"
        "- Slight wording variations that preserve the meaning (e.g., 'United States' vs 'USA' vs 'US')\n"
        "- Additional correct context that doesn't contradict the answer (e.g., 'Paris, France' when answer is 'Paris')\n\n"
        "However, the answer must be factually correct. Do not accept:\n"
        "- Wrong entities (e.g., 'London' when answer is 'Paris')\n"
        "- Wrong numbers or dates\n"
        "- Contradictory information\n"
        "- Incomplete answers that miss key information\n"
        "- Completely unrelated content\n\n"
        "IMPORTANT: Respond with ONLY the single word 'true' or 'false'. "
        "Do not include any explanation, punctuation, prefixes like 'Response:', or additional text. "
        "Output exactly one word: either true or false.\n\n"
        "Examples:\n\n"
        "Example 1 - Equivalent entities:\n"
        "Question: Which country has the largest population?\n"
        "Correct Answer: China\n"
        "Proposed Answer: People's Republic of China\n"
        "Response: true\n\n"
        "Example 2 - Additional correct context:\n"
        "Question: Where is the Eiffel Tower located?\n"
        "Correct Answer: Paris\n"
        "Proposed Answer: Paris, France\n"
        "Response: true\n\n"
        "Example 3 - Wrong entity:\n"
        "Question: Who wrote Romeo and Juliet?\n"
        "Correct Answer: Shakespeare\n"
        "Proposed Answer: Charles Dickens\n"
        "Response: false\n\n"
        "Example 4 - Missing critical information:\n"
        "Question: When did World War II end?\n"
        "Correct Answer: September 2, 1945\n"
        "Proposed Answer: 1945\n"
        "Response: false\n\n"
        "Example 5 - Completely unrelated content:\n"
        "Question: What is the capital of Japan?\n"
        "Correct Answer: Tokyo\n"
        "Proposed Answer: I enjoy watching movies on weekends.\n"
        "Response: false\n\n"
        "Remember: Output ONLY 'true' or 'false' with no other text."
    )
    return _judge_answer_with_llm(
        ground_truth,
        final_answer,
        question,
        JUDGE_MODEL,
        system_message,
        "Plaintext Judge System"
    )


def judge_answer(ground_truth: str, final_answer: str, question: str, dataset_type: str) -> bool:
    """Judge answer based on dataset type."""
    # AIME uses the same mathematical judging as MATH
    if dataset_type == "aime":
        dataset_type = "math"
    
    if dataset_type == "gpqa":
        return judge_mcq_answer(ground_truth, final_answer, question)
    elif dataset_type == "math":
        return judge_math_answer(ground_truth, final_answer, question)
    elif dataset_type == "musique":
        return judge_plaintext_answer(ground_truth, final_answer, question)
    elif dataset_type == "mmlupro":
        return judge_mmlupro_answer(ground_truth, final_answer, question)
    else:
        raise ValueError(f'Unsupported dataset_type: {dataset_type}')


def are_answers_equivalent(answer1: str, answer2: str, question: str = "", dataset_type: Optional[str] = None) -> bool:
    """
    Check if two answers are mathematically or factually equivalent using LLM judge with caching.
    
    Args:
        answer1: The ground truth answer
        answer2: The prediction answer to check
        question: The question being answered
        dataset_type: Optional dataset type for context
    """
    if not isinstance(answer1, str):
        answer1 = str(answer1)
    if not isinstance(answer2, str):
        answer2 = str(answer2)
    
    # Create cache key
    cache_key = _create_cache_key(answer1, answer2, question, dataset_type)
    
    # Check cache - now returns tuple of (result, stored_question)
    if cache_key in _equivalence_cache:
        cached_result, _ = _equivalence_cache[cache_key]
        return cached_result
    
    # Quick exact match first
    if answer1.strip().lower() == answer2.strip().lower():
        # Cache the result with question
        _equivalence_cache[cache_key] = (True, question)
        return True
    
    # Use LLM judge to check equivalence
    try:
        # Check if answer2 is equivalent to answer1
        # Pass answer1 as ground truth and answer2 as prediction to judge_answer
        is_equivalent = judge_answer(answer1, answer2, question or "Answer equivalence check", dataset_type)
        
        # Cache the result with question text
        _equivalence_cache[cache_key] = (is_equivalent, question)
        
        return is_equivalent
    except Exception as e:
        print(f"Error in LLM equivalence check: {e}")
        # Fallback to simple string comparison
        return answer1.strip().lower() == answer2.strip().lower()


def clear_equivalence_cache():
    """Clear the equivalence cache if needed."""
    global _equivalence_cache
    _equivalence_cache.clear()


def get_cache_stats():
    """Get cache statistics for monitoring."""
    # Count unique questions (using hash as identifier since question might be very long)
    unique_question_hashes = set()
    for key, _ in _equivalence_cache.items():
        _, _, question_hash, _ = key
        unique_question_hashes.add(question_hash)
    
    return {
        "cache_size": len(_equivalence_cache),
        "cached_pairs": len(_equivalence_cache),
        "unique_questions": len(unique_question_hashes)
    }


def dump_equivalence_cache(filepath: str = "equivalence_cache_dump.json"):
    """Dump the equivalence cache to a JSON file for analysis."""
    import json
    from datetime import datetime
    
    # Convert cache to a more readable format
    cache_dump = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_entries": len(_equivalence_cache),
            "unique_pairs": len(_equivalence_cache)  # Each entry is unique due to hashing
        },
        "entries": []
    }
    
    # Convert tuple keys to readable format
    for key, (is_correct, question_text) in _equivalence_cache.items():
        norm_answer1, norm_answer2, question_hash, dataset_type = key
        
        cache_dump["entries"].append({
            "answer1_normalized": norm_answer1,  # Full answer, no truncation
            "answer2_normalized": norm_answer2,  # Full answer, no truncation
            "question_hash": question_hash,
            "question_full": question_text,  # Full question text, no truncation
            "dataset_type": dataset_type,
            "judge_result": is_correct
        })
    
    # Sort by judge result and dataset for easier analysis
    cache_dump["entries"].sort(key=lambda x: (x["judge_result"], x["dataset_type"] or ""))
    
    with open(filepath, 'w') as f:
        json.dump(cache_dump, f, indent=2, ensure_ascii=False)  # ensure_ascii=False to preserve unicode
    
    print(f"Equivalence cache dumped to {filepath}")
    print(f"Total cache entries: {len(_equivalence_cache)}")
    
    # Print summary statistics
    correct_count = sum(1 for _, (v, _) in _equivalence_cache.items() if v)
    incorrect_count = len(_equivalence_cache) - correct_count
    print(f"Correct judgments: {correct_count}")
    print(f"Incorrect judgments: {incorrect_count}")
    
    # Dataset breakdown
    dataset_counts = {}
    for key, _ in _equivalence_cache.items():
        dataset = key[3] or "unknown"
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print("\nCache entries by dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")
    
    return cache_dump
