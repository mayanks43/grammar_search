"""
LLM interface for cluster evolution system with simple worker token budgets and task rejection.
Each worker gets a model-specific token budget per minute with rejection for oversized tasks.
"""

import json
import time
import random
import openai
import logging
import os
import re
import threading
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from common.config import (
    AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, RETRIES,
    USE_OPENAI, WORKER_TOKEN_CONFIG, MODEL_CAPACITY
)
from common.debug_logger import debug_logger
from common.azure_utils import azure_retry_backoff
from common.token_tracker import token_tracker, estimate_tokens
from collections import defaultdict


# Custom exception for token budget rejections
class TokenBudgetExceededException(Exception):
    """Raised when a task exceeds worker token budget after multiple attempts."""
    pass


# Directory for rejected task logs
REJECTED_TASKS_DIR = Path("rejected_token_budget_tasks")
REJECTED_TASKS_DIR.mkdir(exist_ok=True)


def save_rejected_task_log(task_info: Dict, tokens_needed: int, model: str, 
                          budget_per_minute: int, attempt_count: int):
    """Save rejected task details to individual file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        thread_id = threading.get_ident()
        filename = f"rejected_task_{timestamp}_thread_{thread_id}.json"
        filepath = REJECTED_TASKS_DIR / filename
        
        rejection_log = {
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id,
            "model": model,
            "tokens_needed": tokens_needed,
            "worker_budget_per_minute": budget_per_minute,
            "attempt_count": attempt_count,
            "rejection_reason": f"Task needs {tokens_needed:,} tokens but worker budget is {budget_per_minute:,}",
            "task_details": task_info,
            "worker_token_config": WORKER_TOKEN_CONFIG.copy()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rejection_log, f, indent=2, ensure_ascii=False)
        
        print(f"🚫 REJECTED TASK logged to: {filepath}")
        
    except Exception as e:
        print(f"Failed to save rejected task log (non-critical): {e}")


class WorkerTokenBudget:
    """Per-worker token budget with model-specific limits that reset every minute."""
    
    def __init__(self):
        self._local = threading.local()
        self.max_rejection_attempts = 3  # Try 3 times (3 minutes) before rejecting
    
    def _get_worker_data(self):
        """Get or initialize thread-local worker data including per-model budgets."""
        if not hasattr(self._local, 'budget_data'):
            self._local.budget_data = {
                # Per-model token tracking
                'model_tokens_used': {},  # model -> tokens_used
                'last_reset': time.time(),
                'rejection_attempts': {},  # task_id -> attempt_count
                # Thread-local response stats per model
                'response_stats': defaultdict(lambda: {
                    'total_tokens': 0,
                    'call_count': 0,
                    'avg_tokens': 0,
                    'max_tokens': 0
                })
            }
        return self._local.budget_data
    
    def _get_model_budget(self, model: str) -> int:
        """Get the token budget for a specific model."""
        model_budgets = WORKER_TOKEN_CONFIG.get('model_budgets', {})
        return model_budgets.get(model, model_budgets.get('default', 100_000))
    
    def _get_model_alert_threshold(self, model: str) -> int:
        """Get the alert threshold for a specific model."""
        alert_thresholds = WORKER_TOKEN_CONFIG.get('large_token_alert_thresholds', {})
        return alert_thresholds.get(model, alert_thresholds.get('default', 80_000))
    
    def _get_estimated_response_tokens(self, model: str, max_tokens: int) -> int:
        """
        Get estimated response tokens based on this worker's historical data.
        Starts conservatively at 0, then uses rolling average.
        Thread-local, no locks needed.
        """
        worker_data = self._get_worker_data()
        stats = worker_data['response_stats'][model]
        
        if stats['call_count'] > 0:
            # Use average from previous responses, capped by max_tokens
            avg = stats['avg_tokens']
            # Add 20% buffer for safety
            estimated = int(avg * 1.2)
            return min(estimated, max_tokens)
        else:
            # First call for this model by this worker - start optimistically with 0
            return 0
    
    def update_response_stats(self, model: str, actual_tokens: int):
        """Update response token statistics after receiving a response. Thread-local, no locks."""
        worker_data = self._get_worker_data()
        stats = worker_data['response_stats'][model]
        
        stats['total_tokens'] += actual_tokens
        stats['call_count'] += 1
        stats['avg_tokens'] = stats['total_tokens'] / stats['call_count']
        stats['max_tokens'] = max(stats['max_tokens'], actual_tokens)
    
    def can_use_tokens(self, tokens_needed: int, model: str, task_info: Dict = None) -> bool:
        """Check if worker can use the requested tokens with model-specific budgets."""
        worker_data = self._get_worker_data()
        current_time = time.time()
        
        # Get model-specific budget
        model_budget = self._get_model_budget(model)
        alert_threshold = self._get_model_alert_threshold(model)
        
        # First, check if task is impossibly large for this model
        if tokens_needed > model_budget:
            # Task will never fit - reject immediately
            save_rejected_task_log(
                task_info or {"model": model, "tokens_needed": tokens_needed},
                tokens_needed,
                model,
                model_budget,
                0  # 0 attempts - immediately rejected
            )
            
            raise TokenBudgetExceededException(
                f"Task needs {tokens_needed:,} tokens but {model} budget is only "
                f"{model_budget:,} tokens per minute"
            )
        
        # Reset budget if a minute has passed
        if current_time - worker_data['last_reset'] >= 60.0:
            worker_data['model_tokens_used'] = {}  # Reset all model budgets
            worker_data['last_reset'] = current_time
            # Clear ALL rejection attempts on budget reset
            worker_data['rejection_attempts'].clear()
        
        # Initialize model token count if needed
        if model not in worker_data['model_tokens_used']:
            worker_data['model_tokens_used'][model] = 0
        
        # Check large token alert
        if tokens_needed > alert_threshold:
            print(f"⚠️  LARGE TOKEN ALERT: {tokens_needed:,} tokens requested for {model} "
                  f"(alert threshold: {alert_threshold:,})")
        
        # Check if we have budget for this model
        model_tokens_used = worker_data['model_tokens_used'][model]
        if model_tokens_used + tokens_needed <= model_budget:
            worker_data['model_tokens_used'][model] += tokens_needed
            return True
        
        # Budget insufficient - track rejection attempts
        task_id = str(hash(str(task_info) + model)) if task_info else f"unknown_{tokens_needed}_{model}"
        
        if task_id not in worker_data['rejection_attempts']:
            worker_data['rejection_attempts'][task_id] = 0
        
        worker_data['rejection_attempts'][task_id] += 1
        
        if worker_data['rejection_attempts'][task_id] >= self.max_rejection_attempts:
            # Save rejection log
            save_rejected_task_log(
                task_info or {"model": model, "tokens_needed": tokens_needed},
                tokens_needed,
                model,
                model_budget,
                worker_data['rejection_attempts'][task_id]
            )
            
            # Clean up this task
            del worker_data['rejection_attempts'][task_id]
            
            # Raise rejection exception
            raise TokenBudgetExceededException(
                f"Task needs {tokens_needed:,} tokens but {model} worker budget is "
                f"{model_budget:,} (after {self.max_rejection_attempts} attempts)"
            )
        
        return False
    
    def wait_for_budget_reset(self):
        """Wait until the next minute for budget reset. Thread-local, no lock needed."""
        worker_data = self._get_worker_data()
        current_time = time.time()
        time_until_reset = 60.0 - (current_time - worker_data['last_reset'])
        
        if time_until_reset > 0:
            logging.info(f"Worker budget exhausted, waiting {time_until_reset:.1f}s for reset")
            time.sleep(time_until_reset)
    
    def get_model_usage_summary(self) -> Dict[str, Dict]:
        """Get current usage summary for all models by this worker."""
        worker_data = self._get_worker_data()
        summary = {}
        
        for model in worker_data.get('model_tokens_used', {}):
            model_budget = self._get_model_budget(model)
            tokens_used = worker_data['model_tokens_used'][model]
            
            summary[model] = {
                'tokens_used': tokens_used,
                'budget': model_budget,
                'utilization': (tokens_used / model_budget) * 100 if model_budget > 0 else 0,
                'remaining': model_budget - tokens_used
            }
        
        return summary


# Global worker budget manager
worker_budget = WorkerTokenBudget()


def calculate_max_workers_for_model(model: str) -> int:
    """Calculate maximum workers for a model based on capacity and per-model worker budget."""
    capacity = MODEL_CAPACITY.get(model, MODEL_CAPACITY["default"])
    
    # Get model-specific budget
    model_budgets = WORKER_TOKEN_CONFIG.get('model_budgets', {})
    tokens_per_worker = model_budgets.get(model, model_budgets.get('default', 100_000))
    
    return max(1, capacity // tokens_per_worker)


def get_openai_api_key(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    openai_file_path = os.path.join(script_dir, path)

    try:
        with open(openai_file_path, 'r') as f:
            first_line = f.readline().strip('\n')
            return first_line
    except FileNotFoundError:
        print(f"Error: The file '{openai_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def get_azure_config():
    """Load Azure configuration from environment or config file."""
    import os
    global AZURE_API_KEY, AZURE_ENDPOINT
    
    if AZURE_API_KEY is None:
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            azure_file_path = os.path.join(script_dir, '/workspace/azure')
            
            config = {}
            with open(azure_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            
            AZURE_API_KEY = config.get('AZURE_OPENAI_API_KEY') or os.getenv("AZURE_OPENAI_API_KEY")
            AZURE_ENDPOINT = config.get('AZURE_OPENAI_ENDPOINT') or os.getenv("AZURE_OPENAI_ENDPOINT")
        except:
            AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
            AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


def get_backbone_client():
    """Initialize the client for backbone models (agent execution)."""
    if USE_OPENAI:
        OPENAI_API_KEY = get_openai_api_key('/workspace/openai')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=180.0
        )
    else:
        get_azure_config()
        return openai.AzureOpenAI(
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            timeout=180.0,
        )


def get_agent_client():
    """Initialize the client for agent models (evolution/reflection)."""
    if USE_OPENAI:
        OPENAI_API_KEY = get_openai_api_key('/workspace/openai')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=180.0
        )
    else:
        get_azure_config()
        return openai.AzureOpenAI(
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            timeout=180.0
        )


def get_judge_client():
    """Initialize the client for the judge model."""
    if USE_OPENAI:
        OPENAI_API_KEY = get_openai_api_key('/workspace/openai')
        return openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=60.0
        )
    else:
        get_azure_config()
        return openai.AzureOpenAI(
            api_version=AZURE_API_VERSION,
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            timeout=60.0
        )


def _estimate_tokens_for_messages(messages: List[Dict], model: str) -> int:
    """Estimate total token count for a list of messages using model-specific encoding."""
    total_text = ""
    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")
        # Add role and formatting overhead
        total_text += f"{role}: {content}\n"
    
    return estimate_tokens(total_text, model)


def _check_worker_token_budget(model: str, messages: List[Dict], max_tokens: int, task_info: Dict = None) -> bool:
    """
    Check and wait for worker token budget if needed with task rejection and usage tracking.
    
    Args:
        model: Model name
        messages: Messages for token estimation
        max_tokens: Maximum output tokens
        task_info: Task information for rejection logging
        
    Returns:
        True if tokens can be used
        
    Raises:
        TokenBudgetExceededException: If task is rejected after max attempts
    """
    input_tokens = _estimate_tokens_for_messages(messages, model)
    
    # Get estimated response tokens based on history
    estimated_response_tokens = worker_budget._get_estimated_response_tokens(model, max_tokens)
    total_tokens_needed = input_tokens + estimated_response_tokens
    
    # Prepare task info for potential rejection logging (full content, no truncation)
    task_context = task_info or {
        "input_tokens_estimated": input_tokens,
        "max_output_tokens": max_tokens,
        "estimated_response_tokens": estimated_response_tokens,
        "total_tokens_needed": total_tokens_needed,
        "messages_full": [
            {
                "role": msg.get("role", ""),
                "content": msg.get("content", "")  # Full content, no truncation
            }
            for msg in messages  # All messages, not just first 2
        ]
    }
    
    # Try to use tokens from worker budget with rejection
    while not worker_budget.can_use_tokens(total_tokens_needed, model, task_context):
        # Budget exhausted, wait for reset (rejection exception may be raised)
        worker_budget.wait_for_budget_reset()
    
    # Record successful token usage for tracking
    from common.worker_usage_tracker import record_worker_token_usage
    record_worker_token_usage(model, total_tokens_needed)
    
    return True


@azure_retry_backoff(max_tries=RETRIES)
def get_text_response_from_gpt(
    msg: str,
    model: str,
    system_message: str,
    temperature: float = 0.5,
    problem_info: str = None,
    agent_info: str = None
) -> str:
    """Get text response from GPT without JSON formatting."""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_message}, 
        {"role": "user", "content": msg}
    ]
    max_tokens = 8192
    
    # Prepare task info for potential rejection
    task_info = {
        "function": "get_text_response_from_gpt",
        "problem_info": problem_info,
        "agent_info": agent_info,
        "temperature": temperature,
        "model": model,
        "system_message_length": len(system_message),
        "user_message_length": len(msg)
    }
    
    try:
        # Check worker token budget with adaptive estimation
        _check_worker_token_budget(model, messages, max_tokens, task_info)
    except TokenBudgetExceededException as e:
        # Task rejected - raise the exception as in original
        print(f"🚫 Task rejected: {e}")
        raise
    
    # Add random sleep of 0-2000ms before call (existing logic)
    time.sleep(random.uniform(0, 2.0))

    client = get_backbone_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None
        # Note: No response_format parameter
    )
    
    response_text = response.choices[0].message.content
    
    if response_text is None or response_text.strip() == "":
        response_text = "ERROR: Empty response from model"
    
    if response.choices[0].finish_reason == "length":
        logging.warning(f"Response truncated at {max_tokens} tokens for {model}")
        # Raise ValueError to trigger retry with backoff
        raise ValueError(f"Response truncated due to length limit ({max_tokens} tokens)")

    # Track actual token usage and update adaptive estimation
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        
        # Update adaptive response estimation
        worker_budget.update_response_stats(model, usage.completion_tokens)
        
        # Track costs
        token_tracker.add_usage(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            call_type="agent_execution"
        )
    
    # Debug logging
    debug_logger.log_llm_call(
        call_type="agent_execution",
        model=model,
        system_prompt=system_message,
        user_prompt=msg,
        response=response_text,
        problem_info=problem_info,
        agent_info=agent_info
    )
    
    return response_text


# Updated get_json_response_from_gpt with adaptive token tracking
@azure_retry_backoff(max_tries=RETRIES) 
def get_json_response_from_gpt(
    msg: str,
    model: str,
    system_message: str,
    temperature: float = 0.5, 
    problem_info: str = None,
    agent_info: str = None
) -> Dict:
    """Get JSON response from GPT with simple worker token budgets and adaptive estimation."""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_message}, 
        {"role": "user", "content": msg}
    ]
    max_tokens = 4096
    
    # Prepare task info for potential rejection
    task_info = {
        "function": "get_json_response_from_gpt",
        "problem_info": problem_info,
        "agent_info": agent_info,
        "temperature": temperature,
        "model": model,
        "system_message_length": len(system_message),
        "user_message_length": len(msg)
    }
    
    try:
        # Check worker token budget with adaptive estimation
        _check_worker_token_budget(model, messages, max_tokens, task_info)
    except TokenBudgetExceededException as e:
        # Task rejected - return error response instead of failing
        print(f"🚫 Task rejected: {e}")
        return {
            "error": "TOKEN_BUDGET_EXCEEDED",
            "message": str(e),
            "rejected": True
        }
    
    # Add random sleep of 0-2000ms before call (existing logic)
    time.sleep(random.uniform(0, 2.0))

    client = get_backbone_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
        response_format={"type": "json_object"}
    )
    
    response_text = response.choices[0].message.content
    
    # Track actual token usage and update adaptive estimation
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        
        # Update adaptive response estimation
        worker_budget.update_response_stats(model, usage.completion_tokens)
        
        # Track costs
        token_tracker.add_usage(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            call_type="agent_execution"
        )
    
    # Debug logging
    debug_logger.log_llm_call(
        call_type="agent_execution",
        model=model,
        system_prompt=system_message,
        user_prompt=msg,
        response=response_text,
        problem_info=problem_info,
        agent_info=agent_info
    )
    
    return json.loads(response_text)


# Updated get_json_response_from_gpt_reflect with adaptive token tracking
@azure_retry_backoff(max_tries=RETRIES)
def get_json_response_from_gpt_reflect(msg_list: List[Dict], model: str, temperature: float = 0.8) -> Dict:
    """Get JSON response for reflection/refinement with simple worker token budgets and adaptive estimation."""
    
    max_tokens = 16384
    
    # Prepare task info for potential rejection
    task_info = {
        "function": "get_json_response_from_gpt_reflect",
        "model": model,
        "temperature": temperature,
        "message_count": len(msg_list),
        "total_content_length": sum(len(msg.get("content", "")) for msg in msg_list)
    }
    
    try:
        # Check worker token budget with adaptive estimation
        _check_worker_token_budget(model, msg_list, max_tokens, task_info)
    except TokenBudgetExceededException as e:
        # Task rejected - return error response instead of failing
        print(f"🚫 Evolution task rejected: {e}")
        return {
            "error": "TOKEN_BUDGET_EXCEEDED",
            "message": str(e),
            "rejected": True
        }

    # Add random sleep of 0-2000ms before call (existing logic)
    time.sleep(random.uniform(0, 2.0))

    client = get_agent_client()
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
        response_format={"type": "json_object"}
    )
    
    response_text = response.choices[0].message.content
    
    # Track actual token usage and update adaptive estimation
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        
        # Update adaptive response estimation
        worker_budget.update_response_stats(model, usage.completion_tokens)
        
        # Track costs
        token_tracker.add_usage(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            call_type="evolution"
        )
    
    # Simplified debug logging - just use last two messages
    system_message = msg_list[-2]['content'] if len(msg_list) >= 2 else ""
    user_message = msg_list[-1]['content'] if len(msg_list) >= 1 else ""
    
    debug_logger.log_llm_call(
        call_type="evolution",
        model=model,
        system_prompt=system_message,
        user_prompt=user_message,
        response=response_text,
        problem_info="Architecture Evolution",
        agent_info="Evolution System"
    )
    
    return json.loads(response_text)


# Updated _judge_answer_with_llm with plaintext true/false responses
@azure_retry_backoff(max_tries=RETRIES)
def _judge_answer_with_llm(
    ground_truth: str,
    final_answer: str,
    question: str, 
    model: str,
    system_message: str,
    agent_info: str
) -> bool:
    """Generic LLM-based answer judging with simple worker token budgets and adaptive estimation."""
    
    # Check for empty answer first
    if not final_answer or not final_answer.strip():
        return False

    user_message = f"Question: {question}\n\nCorrect Answer: {ground_truth}\n\nProposed Answer: {final_answer}"
    messages = [
        {"role": "system", "content": system_message}, 
        {"role": "user", "content": user_message}
    ]
    max_tokens = 4096

    # Prepare task info for potential rejection
    task_info = {
        "function": "_judge_answer_with_llm",
        "model": model,
        "agent_info": agent_info,
        "question_length": len(question),
        "ground_truth_length": len(ground_truth),
        "final_answer_length": len(str(final_answer))
    }
    
    # Handle token budget separately
    try:
        # Check worker token budget with adaptive estimation
        _check_worker_token_budget(model, messages, max_tokens, task_info)
    except TokenBudgetExceededException as e:
        # Task rejected - return False for judging (conservative approach)
        print(f"🚫 Judging task rejected: {e}")
        return False
    
    # Add random sleep of 0-2000ms before judge call (existing logic)
    time.sleep(random.uniform(0, 2.0))

    # Make the API call
    client = get_judge_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        stop=None
        # Note: No response_format parameter - expecting plaintext
    )
    
    response_text = response.choices[0].message.content
    
    # Track tokens and update adaptive estimation (non-critical, so wrapped in try-except)
    try:
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            
            # Update adaptive response estimation
            worker_budget.update_response_stats(model, usage.completion_tokens)
            
            # Track costs
            token_tracker.add_usage(
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                call_type="judging"
            )
    except Exception as e:
        logging.debug(f"Token tracking failed (non-critical): {e}")
    
    # Debug logging (non-critical, so wrapped in try-except)
    try:
        debug_logger.log_llm_call(
            call_type="judging",
            model=model,
            system_prompt=system_message,
            user_prompt=user_message,
            response=response_text,
            problem_info=f"Ground Truth: {ground_truth}",
            agent_info=agent_info
        )
    except Exception as e:
        logging.debug(f"Debug logging failed (non-critical): {e}")
    
    # NOW PARSE THE RESPONSE - NO TRY-EXCEPT HERE SO ERRORS BUBBLE UP TO DECORATOR
    # Parse response - expecting exactly "true" or "false"
    response_text_stripped = response_text.strip() if response_text else ""
    
    # Check for empty response first
    if not response_text_stripped:
        # This ValueError will bubble up to the decorator for retry
        logging.warning(f"Empty response from judge {model}. Triggering retry...")
        raise ValueError(f"Invalid judge response: expected 'true' or 'false', got empty response")
    
    # STRICT parsing - only accept exactly "true" or "false" (case-insensitive)
    response_lower = response_text_stripped.lower()
    
    if response_lower == "true":
        return True
    elif response_lower == "false":
        return False
    else:
        # ANY other response is invalid - this ValueError will bubble up to the decorator for retry
        response_preview = response_text[:100] if len(response_text) > 100 else response_text
        logging.warning(f"Invalid judge response from {model}: '{response_preview}'")
        raise ValueError(f"Invalid judge response: expected exactly 'true' or 'false', got '{response_preview}'")


# Print worker configuration on module load
def print_worker_config():
    """Print worker configuration for all models."""
    print(f"\n🔧 Per-Model Worker Token Budget Configuration:")
    print(f"   Max rejection attempts: {worker_budget.max_rejection_attempts} (before task rejection)")
    print(f"   Rejected task logs directory: {REJECTED_TASKS_DIR}")
    
    model_budgets = WORKER_TOKEN_CONFIG.get('model_budgets', {})
    alert_thresholds = WORKER_TOKEN_CONFIG.get('large_token_alert_thresholds', {})
    
    print(f"\n📊 Token Budgets and Max Workers per Model:")
    
    # Show configured models
    all_models = set(list(MODEL_CAPACITY.keys()) + list(model_budgets.keys()))
    all_models.discard('default')
    
    for model in sorted(all_models):
        capacity = MODEL_CAPACITY.get(model, MODEL_CAPACITY["default"])
        budget = model_budgets.get(model, model_budgets.get('default', 100_000))
        alert = alert_thresholds.get(model, alert_thresholds.get('default', 80_000))
        max_workers = calculate_max_workers_for_model(model)
        
        print(f"\n   {model}:")
        print(f"      Token budget per worker: {budget:,} tokens/minute")
        print(f"      Alert threshold: {alert:,} tokens")
        print(f"      Total capacity: {capacity:,} tokens/minute")
        print(f"      Max workers: {max_workers} ({capacity:,} / {budget:,})")
    
    # Show default configuration
    if 'default' in model_budgets:
        print(f"\n   Default (unknown models):")
        print(f"      Token budget per worker: {model_budgets['default']:,} tokens/minute")
        print(f"      Alert threshold: {alert_thresholds.get('default', 80_000):,} tokens")

print_worker_config()
