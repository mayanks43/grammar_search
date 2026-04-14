"""
Worker token usage tracker to monitor actual token consumption patterns.
Tracks per-worker session rates with individual timing.
FIXED: Uses write-only locks and per-worker session tracking to prevent deadlocks.
"""

import threading
import time
from typing import Dict, List
from collections import defaultdict
import statistics


class WorkerUsageTracker:
    """Tracks actual worker token usage patterns by model with per-worker session timing."""
    
    def __init__(self):
        self._write_lock = threading.Lock()  # Only for writes
        # Track per-worker sessions: model -> worker_id -> session_data
        self._worker_sessions = defaultdict(lambda: defaultdict(lambda: {
            "start_time": None,
            "total_tokens": 0,
            "last_activity": None,
            "call_count": 0
        }))
        self._stage_summaries = []  # Track summaries for each stage
    
    def record_token_usage(self, model: str, worker_id: int, tokens_used: int):
        """Record token usage for a specific worker."""
        current_time = time.time()
        
        with self._write_lock:  # Only lock for writes
            session = self._worker_sessions[model][worker_id]
            
            # Initialize worker session if first time
            if session["start_time"] is None:
                session["start_time"] = current_time
            
            session["total_tokens"] += tokens_used
            session["last_activity"] = current_time
            session["call_count"] += 1
    
    def get_worker_usage_per_minute(self, model: str, worker_id: int) -> int:
        """Get tokens per minute for a specific worker - READ ONLY."""
        # NO LOCK - just read the data structure
        session = self._worker_sessions[model][worker_id]
        
        if session["total_tokens"] == 0 or session["start_time"] is None or session["last_activity"] is None:
            return 0
        
        # Calculate this worker's personal rate based on their session duration
        worker_elapsed_minutes = (session["last_activity"] - session["start_time"]) / 60
        if worker_elapsed_minutes <= 0:
            return 0
        
        return int(session["total_tokens"] / worker_elapsed_minutes)
    
    def get_model_usage_stats(self, model: str) -> Dict:
        """Get usage statistics for all workers of a model - READ ONLY."""
        # NO LOCK - just read current state
        if model not in self._worker_sessions:
            return {
                "active_workers": 0,
                "max_tokens_per_minute": 0,
                "avg_tokens_per_minute": 0,
                "min_tokens_per_minute": 0,
                "worker_utilization": [],
                "total_tokens_across_workers": 0,
                "total_calls": 0
            }
        
        worker_rates = []
        total_tokens = 0
        total_calls = 0
        active_workers = 0
        
        # Read without lock - might get slightly inconsistent view but that's fine for monitoring
        for worker_id, session in self._worker_sessions[model].items():
            if session["total_tokens"] > 0 and session["start_time"] is not None and session["last_activity"] is not None:
                # Calculate this worker's personal rate
                worker_elapsed_minutes = (session["last_activity"] - session["start_time"]) / 60
                if worker_elapsed_minutes > 0:
                    worker_rate = session["total_tokens"] / worker_elapsed_minutes
                    worker_rates.append(worker_rate)
                    active_workers += 1
                
                total_tokens += session["total_tokens"]
                total_calls += session["call_count"]
        
        if not worker_rates:
            return {
                "active_workers": 0,
                "max_tokens_per_minute": 0,
                "avg_tokens_per_minute": 0,
                "min_tokens_per_minute": 0,
                "worker_utilization": [],
                "total_tokens_across_workers": 0,
                "total_calls": 0
            }
        
        return {
            "active_workers": active_workers,
            "max_tokens_per_minute": max(worker_rates),
            "avg_tokens_per_minute": statistics.mean(worker_rates),
            "min_tokens_per_minute": min(worker_rates),
            "worker_utilization": worker_rates,
            "total_tokens_across_workers": total_tokens,
            "total_calls": total_calls
        }
    
    def _get_model_budget(self, model: str) -> int:
        """Get the token budget for a specific model from config."""
        from common.config import WORKER_TOKEN_CONFIG        

        model_budgets = WORKER_TOKEN_CONFIG.get('model_budgets', {})
        return model_budgets.get(model, model_budgets.get('default', 100_000))
    
    def print_stage_usage_summary(self, stage_name: str, models_used: List[str]):
        """Print usage summary for a completed stage - READ ONLY for stats."""
        print(f"\n📊 WORKER TOKEN USAGE SUMMARY - {stage_name.upper()}")
        print("-" * 70)
        
        stage_data = {
            "stage": stage_name,
            "timestamp": time.time(),
            "models": {}
        }
        
        for model in models_used:
            # Lock-free read of stats
            stats = self.get_model_usage_stats(model)
            stage_data["models"][model] = stats
            
            if stats["active_workers"] > 0:
                # Get model-specific budget
                budget_per_worker = self._get_model_budget(model)
                
                max_utilization = (stats["max_tokens_per_minute"] / budget_per_worker) * 100
                avg_utilization = (stats["avg_tokens_per_minute"] / budget_per_worker) * 100
                
                print(f"  {model}:")
                print(f"    Active workers: {stats['active_workers']}")
                print(f"    Total tokens used: {stats['total_tokens_across_workers']:,}")
                print(f"    Total LLM calls: {stats['total_calls']:,}")
                print(f"    Max worker rate: {stats['max_tokens_per_minute']:,.0f} tokens/min ({max_utilization:.1f}% of budget)")
                print(f"    Avg worker rate: {stats['avg_tokens_per_minute']:,.0f} tokens/min ({avg_utilization:.1f}% of budget)")
                print(f"    Min worker rate: {stats['min_tokens_per_minute']:,.0f} tokens/min")
                print(f"    Worker budget: {budget_per_worker:,} tokens/min")
                
                # Show utilization distribution
                if stats["worker_utilization"]:
                    utilization_percentages = [
                        (rate / budget_per_worker) * 100 
                        for rate in stats["worker_utilization"]
                    ]
                    print(f"    Utilization range: {min(utilization_percentages):.1f}% - {max(utilization_percentages):.1f}%")
                    
                    # Efficiency analysis
                    if max_utilization > 90:
                        print(f"    ⚠️  HIGH UTILIZATION: Some workers hitting budget limit")
                    elif avg_utilization < 30:
                        print(f"    💡 LOW UTILIZATION: Workers underutilized on average")
                    else:
                        print(f"    ✅ GOOD UTILIZATION: Workers efficiently used")
                else:
                    print(f"    📊 No utilization data available")
            else:
                print(f"  {model}: No active workers recorded")
        
        # Only lock when writing to stage_summaries
        with self._write_lock:
            self._stage_summaries.append(stage_data)
        
        print("-" * 70)
    
    def get_overall_summary(self) -> Dict:
        """Get overall summary across all stages - READ ONLY."""
        # NO LOCK - read current state
        return {
            "total_stages_tracked": len(self._stage_summaries),
            "stage_summaries": list(self._stage_summaries),  # Create copy for safety
            "current_active_models": list(self._worker_sessions.keys())
        }
    
    def reset_for_new_stage(self, stage_name: str):
        """Optional: Reset tracking for a new stage if you want stage-specific stats."""
        with self._write_lock:
            print(f"🔄 Resetting worker token tracking for {stage_name}")
            self._worker_sessions.clear()


# Global worker usage tracker
worker_usage_tracker = WorkerUsageTracker()


# Function to be called from LLM interface when tokens are used
def record_worker_token_usage(model: str, tokens_used: int):
    """Record token usage for current worker thread."""
    worker_id = threading.get_ident()
    worker_usage_tracker.record_token_usage(model, worker_id, tokens_used)
