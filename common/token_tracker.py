"""
Token tracking for cost monitoring with thread-safe aggregation.
Fixed to properly aggregate costs from worker threads.
"""

import threading
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path
from datetime import datetime

# Model pricing per 1M tokens (in USD)
MODEL_PRICING = {
    "gpt-4.1-nano": {
        "input": 0.1, 
        "output": 0.4, 
        "context_window": 1_047_576,
    },
    "gpt-4o-mini": {
        "input": 0.15, 
        "output": 0.6, 
        "context_window": 128_000,
    },
    "gpt-4.1-mini": {
        "input": 0.4, 
        "output": 1.6, 
        "context_window": 1_047_576,
    }, 
    "gpt-4.1": {
        "input": 2.0, 
        "output": 8.0, 
        "context_window": 1_047_576,
    },
    "gpt-5": {
        "input": 1.25,
        "output": 10.0,
        "context_window": 400_000,
    },
    "openai/gpt-oss-20b": {
        "input": 0.03,
        "output": 0.15,
        "context_window": 131_072,
    },
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.6,
        "context_window": 131_072,
    },
    "google/gemma-3-27b-it": {
        "input": 0.09,
        "output": 0.16,
        "context_window": 131_072,
    },
    "google/gemma-3-12b-it": {
        "input": 0.05,
        "output": 0.10,
        "context_window": 131_072,
    },
    # Add fallback for unknown models
    "default": {
        "input": 1.0, 
        "output": 4.0, 
        "context_window": 1_000_000,
    }
}

# Alert threshold (90% of context window)
CONTEXT_ALERT_THRESHOLD = 0.9

# Token estimation using tiktoken with model-specific encodings
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - falling back to rough estimation")


def estimate_tokens(text: str, model: str) -> int:
    """
    Estimate token count for text using tiktoken with model-specific encoding.
    """
    if not TIKTOKEN_AVAILABLE:
        # Rough fallback: ~4 characters per token
        return len(text) // 4

    try:
        # Use tiktoken's encoding_for_model for accurate model-specific tokenization
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
        
    except KeyError:
        # Model not found in tiktoken, try common encodings
        try:
            if "gpt-4" in model.lower():
                encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            elif "gpt-3.5" in model.lower():
                encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 encoding  
            else:
                encoding = tiktoken.get_encoding("cl100k_base")  # Default to GPT-4 encoding
            
            return len(encoding.encode(text))
            
        except Exception as e:
            logging.warning(f"Token estimation failed for {model}: {e}")
            # Fallback to character-based estimation
            return len(text) // 4
    
    except Exception as e:
        logging.warning(f"Token estimation failed for {model}: {e}")
        # Fallback to character-based estimation
        return len(text) // 4


def format_cost(cost: float, is_cached: bool = False) -> str:
    """
    Format cost with smart decimal places.
    """
    if is_cached:
        return "$0.00 (cached)"
    
    # Smart decimal places - show minimum significant digits
    if cost == 0:
        return "$0.00"
    elif cost < 0.00001:
        return f"${cost:.8f}".rstrip('0').rstrip('.')
    elif cost < 0.0001:
        return f"${cost:.6f}".rstrip('0').rstrip('.')
    elif cost < 0.01:
        return f"${cost:.5f}".rstrip('0').rstrip('.')
    elif cost < 1.00:
        return f"${cost:.4f}".rstrip('0').rstrip('.')
    else:
        return f"${cost:.2f}"


class TokenTracker:
    """Thread-safe token and cost tracker with proper aggregation across worker threads."""
    
    def __init__(self):
        self._local = threading.local()
        self._global_lock = threading.Lock()
        self._aggregated_costs = {}  # Aggregated costs from all threads
    
    def _get_thread_data(self):
        """Get or initialize thread-local data."""
        if not hasattr(self._local, 'totals'):
            self._local.totals = {}
        return self._local.totals
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int, 
                  call_type: str = "unknown"):
        """Add token usage from an LLM call (thread-local)."""
        try:
            thread_data = self._get_thread_data()
            
            if model not in thread_data:
                thread_data[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_calls": 0,
                    "total_cost": 0.0
                }
            
            # Get pricing info
            pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
            
            # Calculate cost
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            call_cost = input_cost + output_cost
            
            # Update thread totals
            thread_data[model]["input_tokens"] += input_tokens
            thread_data[model]["output_tokens"] += output_tokens
            thread_data[model]["total_calls"] += 1
            thread_data[model]["total_cost"] += call_cost
            
            # Check for context window alert
            context_limit = pricing["context_window"]
            if input_tokens > context_limit * CONTEXT_ALERT_THRESHOLD:
                print(f"⚠️  ALERT: Input tokens ({input_tokens:,}) approaching context limit "
                      f"({context_limit:,}) for {model} in {call_type}")
            
        except Exception as e:
            print(f"Token tracking error (non-critical): {e}")
    
    def aggregate_thread_costs(self):
        """
        Aggregate costs from current thread into global storage.
        This MUST be called at the end of each worker thread's execution.
        """
        thread_data = self._get_thread_data()
        
        if not thread_data:
            return
        
        # Aggregate into global storage (thread-safe)
        with self._global_lock:
            for model, data in thread_data.items():
                if model not in self._aggregated_costs:
                    self._aggregated_costs[model] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_calls": 0,
                        "total_cost": 0.0
                    }
                
                self._aggregated_costs[model]["input_tokens"] += data["input_tokens"]
                self._aggregated_costs[model]["output_tokens"] += data["output_tokens"]
                self._aggregated_costs[model]["total_calls"] += data["total_calls"]
                self._aggregated_costs[model]["total_cost"] += data["total_cost"]
        
        # Clear thread-local data after aggregating to prevent double-counting
        self._local.totals = {}
    
    def get_aggregated_costs(self) -> Tuple[float, Dict[str, Dict]]:
        """
        Get all aggregated costs from all threads.
        This is the main method for getting total costs across worker threads.
        """
        with self._global_lock:
            # Calculate totals from aggregated data
            total_cost = 0.0
            model_breakdown = {}
            
            for model, data in self._aggregated_costs.items():
                pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
                input_cost = (data["input_tokens"] / 1_000_000) * pricing["input"]
                output_cost = (data["output_tokens"] / 1_000_000) * pricing["output"]
                model_cost = input_cost + output_cost
                
                total_cost += model_cost
                model_breakdown[model] = {
                    "input_tokens": data["input_tokens"],
                    "output_tokens": data["output_tokens"],
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": model_cost,
                    "calls": data["total_calls"]
                }
            
            return total_cost, model_breakdown
    
    def clear_aggregated_costs(self):
        """Clear all aggregated costs (use with caution)."""
        with self._global_lock:
            self._aggregated_costs = {}
    
    def get_total_cost(self) -> float:
        """Get total cost across all threads."""
        total_cost, _ = self.get_aggregated_costs()
        return total_cost
    
    def get_cost_breakdown(self) -> Dict[str, Dict]:
        """Get detailed cost breakdown across all threads."""
        _, breakdown = self.get_aggregated_costs()
        return breakdown
    
    def reset_thread_costs(self) -> Dict[str, Dict]:
        """
        Reset current thread's cost tracking and return the costs before reset.
        This is mainly for backwards compatibility.
        """
        # Get current costs before reset
        thread_data = self._get_thread_data()
        breakdown = {}
        
        for model, data in thread_data.items():
            pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
            
            input_tokens = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            
            breakdown[model] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost,
                "calls": data.get("total_calls", 0)
            }
        
        # Clear thread data
        self._local.totals = {}
        
        return breakdown
    
    def get_cumulative_costs(self) -> Tuple[float, Dict[str, Dict]]:
        """
        Get cumulative costs including all aggregated data.
        This is an alias for get_aggregated_costs for backward compatibility.
        """
        return self.get_aggregated_costs()
    
    def print_summary(self, title: str = "Token Usage Summary"):
        """Print cost summary with enhanced formatting."""
        total_cost, combined_data = self.get_aggregated_costs()
        
        if not combined_data:
            print(f"\n{title}: No LLM calls tracked")
            return
        
        print(f"\n{title}:")
        print("-" * 50)
        
        total_input = 0
        total_output = 0
        total_calls = 0
        
        for model, data in combined_data.items():
            input_tokens = data["input_tokens"]
            output_tokens = data["output_tokens"]
            calls = data["calls"]
            cost = data["total_cost"]
            
            total_input += input_tokens
            total_output += output_tokens
            total_calls += calls
            
            print(f"  {model}:")
            print(f"    Calls: {calls:,}")
            print(f"    Input: {input_tokens:,} tokens ({format_cost(data['input_cost'])})")
            print(f"    Output: {output_tokens:,} tokens ({format_cost(data['output_cost'])})")
            print(f"    Cost: {format_cost(cost)}")
        
        print(f"\n  TOTAL:")
        print(f"    All calls: {total_calls:,}")
        print(f"    All input tokens: {total_input:,}")
        print(f"    All output tokens: {total_output:,}")
        print(f"    Total cost: {format_cost(total_cost)}")
        print("-" * 50)


# Global token tracker instance
token_tracker = TokenTracker()
