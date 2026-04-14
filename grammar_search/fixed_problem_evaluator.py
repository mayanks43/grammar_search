#!/usr/bin/env python3
"""
Fixed Problem Set Evaluator with proper cost tracking across worker threads.
Fixed to aggregate costs from worker threads and return them properly.
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Import from new data module
from common.data_utils import get_validation_examples
from common.execution_utils import ThreadSafeAgentExecutor
from common.answer_equivalence import are_answers_equivalent
from common.llm_interface import calculate_max_workers_for_model
from common.config import BACKBONE_MODEL, AGENT_MODEL, JUDGE_MODEL
from common.performance_stats import calculate_system_performance
from common.worker_usage_tracker import worker_usage_tracker
from common.token_tracker import token_tracker, format_cost

# Progress tracking
from common.rolling_queue_processor import SimpleProgressPrinter

# Import external cache
from grammar_search.external_cache import ExternalEvaluationCache


class FixedProblemSetEvaluator:
    """Evaluates systems on a fixed set of problems with proper cost tracking across threads."""
    
    def __init__(self, dataset_type: str = "gpqa", num_problems: int = 100, 
                 num_runs: int = 4, cache_dir: str = None,
                 method_version: str = "v1", use_external_cache: bool = True):
        """
        Initialize evaluator with external cache support.
        
        Args:
            dataset_type: Dataset name (math, gpqa, aime, musique)
            num_problems: Number of validation problems
            num_runs: Runs per problem
            cache_dir: Directory for external cache (default: ~/.fusion_adas_cache)
            method_version: Version identifier for cache isolation
            use_external_cache: Whether to use external cache (can be disabled)
        """
        self.dataset_type = dataset_type
        self.num_problems = num_problems
        self.num_runs = num_runs
        self.is_test = False  # Set to True when evaluating test set
        
        self.problems = []
        self.evaluation_cache = {}  # In-memory cache (existing behavior)
        
        # Initialize external cache
        self.use_external_cache = use_external_cache
        self.method_version = method_version
        
        if self.use_external_cache:
            self.external_cache = ExternalEvaluationCache(
                cache_dir=cache_dir,
                method_version=method_version
            )
            print(f"External cache enabled with method version: {method_version}")
        else:
            self.external_cache = None
            print("External cache disabled")
    
    def initialize_problem_set(self) -> bool:
        """Initialize or load the fixed problem set using new data utilities."""
        try:
            print(f"Loading {self.num_problems} {self.dataset_type} validation problems...")
            
            # Use new data utilities to get validation problems
            self.problems = get_validation_examples(self.dataset_type, self.num_problems)
            
            if not self.problems:
                print(f"Failed to load problems for {self.dataset_type}")
                return False
            
            print(f"Loaded {len(self.problems)} {self.dataset_type} problems")
            
            # Print sample problems
            print("Sample problems:")
            for i, problem in enumerate(self.problems[:3]):
                truncated_problem = problem["problem"][:100] + "..." if len(problem["problem"]) > 100 else problem["problem"]
                print(f"  {i+1}. {truncated_problem}")
            
            return True
                
        except Exception as e:
            print(f"Failed to initialize problem set: {e}")
            return False
    
    def _evaluate_single_problem_run(self, task_data: tuple) -> Dict:
        """Evaluate system on a single problem run (for parallel execution)."""
        system, problem_idx, problem, run_id = task_data
        
        # Import debug logger
        from common.debug_logger import debug_logger
        
        # Start debug context - use problem index and run for unique identification
        debug_logger.start_problem_evaluation(system["name"], problem_idx * 100 + run_id)
        
        try:
            # Execute system
            exec_result = ThreadSafeAgentExecutor.execute_system_safely(
                system, problem["problem"]
            )
            
            if exec_result["success"]:
                # Judge correctness
                is_correct = are_answers_equivalent(
                    problem["answer"],
                    exec_result["answer"],
                    problem["problem"],
                    self.dataset_type
                )
                
                result = {
                    "problem_idx": problem_idx,
                    "run_id": run_id,
                    "success": True,
                    "is_correct": is_correct,
                    "execution_failed": False,
                    "execution_time": 0.0  # Not tracking detailed timing
                }
            else:
                result = {
                    "problem_idx": problem_idx,
                    "run_id": run_id,
                    "success": False,
                    "is_correct": False,
                    "execution_failed": True,
                    "execution_time": 0.0
                }
            
            # End debug context and write file
            debug_logger.end_problem_evaluation()
            
            # CRITICAL FIX: Aggregate this worker thread's costs before returning
            # This ensures the costs tracked in this thread are saved to shared storage
            token_tracker.aggregate_thread_costs()
            
            return result
                
        except Exception as e:
            # End debug context even on error
            debug_logger.end_problem_evaluation()
            
            # CRITICAL FIX: Aggregate costs even on error
            token_tracker.aggregate_thread_costs()
            
            return {
                "problem_idx": problem_idx,
                "run_id": run_id,
                "success": False,
                "is_correct": False,
                "execution_failed": True,
                "error": str(e),
                "execution_time": 0.0
            }
    
    def evaluate_system(self, system: Dict[str, Any]) -> Tuple[float, float, float, Dict, Dict]:
        """
        Evaluate system on the fixed problem set with proper cost tracking.
        
        Cache check order:
        1. In-memory cache (doesn't count as iteration)
        2. External cache (counts as iteration, updates in-memory)
        3. Actual evaluation (counts as iteration, updates both caches)
        
        Args:
            system: System dictionary with name and code
            
        Returns:
            Tuple of (performance_score, std_error, std_dev, full_results_dict, cost_info_dict)
        """
        system_name = system["name"]
        component_sequence = system.get("component_sequence", [])
        
        # Create in-memory cache key (existing behavior)
        problem_set_hash = hash(json.dumps(self.problems, sort_keys=True))
        memory_cache_key = (system_name, problem_set_hash)
        
        # STEP 1: Check in-memory cache first (existing behavior - doesn't count as iteration)
        if memory_cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[memory_cache_key]
            print(f"In-memory cache hit for {system_name} (duplicate - not counted)")
            # Return cached result with cost info indicating it was cached
            cost_info = {"total_cost": 0.0, "from_cache": True, "model_breakdown": {}}
            # Cached result now has 4 elements: performance_score, std_error, std_dev, full_results
            return cached_result[0], cached_result[1], cached_result[2], cached_result[3], cost_info
        
        # STEP 2: Check external cache (counts as iteration if found)
        if self.use_external_cache and self.external_cache and component_sequence:
            cache_hit, cached_data = self.external_cache.get(
                dataset_type=self.dataset_type,
                is_test=self.is_test,
                num_problems=len(self.problems),
                num_runs=self.num_runs,
                model=BACKBONE_MODEL,
                judge_model=JUDGE_MODEL,
                component_sequence=component_sequence
            )
            
            if cache_hit:
                # Extract results from cached data
                performance_score = cached_data["performance_score"]
                std_error = cached_data["std_error"]
                full_results = cached_data["full_results"]
                
                # Get std_dev from cached data, or calculate it
                std_dev = full_results.get("std_dev", 0.0)
                if std_dev == 0.0 and "run_accuracies" in full_results:
                    run_accuracies = full_results["run_accuracies"]
                    if len(run_accuracies) > 1:
                        std_dev = np.std(run_accuracies, ddof=1)
                    else:
                        std_dev = 0.0
                
                # Update in-memory cache (prevents future duplicate checks)
                self.evaluation_cache[memory_cache_key] = (performance_score, std_error, std_dev, full_results)
                
                # Reset worker usage tracker (for consistency with actual evaluation)
                worker_usage_tracker.reset_for_new_stage(f"System: {system_name} (cached)")
                
                print(f"External cache hit for {system_name} - using cached results (counts as iteration)")
                print(f"Cached performance: {performance_score:.3f} ± {std_error:.3f} (se) ± {std_dev:.3f} (sd)")
                
                # Show cached worker usage summary (empty since no actual work)
                worker_usage_tracker.print_stage_usage_summary(
                    f"System: {system_name} (cached)", 
                    [BACKBONE_MODEL, AGENT_MODEL, JUDGE_MODEL]
                )
                
                # Return with cost info indicating it was cached
                cost_info = {"total_cost": 0.0, "from_cache": True, "model_breakdown": {}}
                return performance_score, std_error, std_dev, full_results, cost_info
        
        # STEP 3: No cache hit - run actual evaluation
        print(f"Cache miss for {system_name} - running full evaluation")
        
        # Reset worker usage tracker before evaluation
        worker_usage_tracker.reset_for_new_stage(f"System: {system_name}")
        
        # Clear aggregated costs before starting new evaluation
        token_tracker.clear_aggregated_costs()
        
        # Prepare all evaluation tasks (problems × runs)
        evaluation_tasks = []
        for problem_idx, problem in enumerate(self.problems):
            for run_id in range(self.num_runs):
                evaluation_tasks.append((system, problem_idx, problem, run_id))
        
        # Execute evaluations in parallel
        max_workers = calculate_max_workers_for_model(BACKBONE_MODEL)
        evaluation_results = []
        execution_failures = 0
        
        # Initialize progress printer
        progress_printer = SimpleProgressPrinter(
            total_tasks=len(evaluation_tasks),
            log_interval=max(10, len(evaluation_tasks) // 20)  # Print 20 times
        )
        
        print(f"Evaluating {system_name} on {len(self.problems)} problems × {self.num_runs} runs = {len(evaluation_tasks)} total evaluations")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_task = {
                executor.submit(self._evaluate_single_problem_run, task_data): task_data
                for task_data in evaluation_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task_data = future_to_task[future]
                try:
                    result = future.result()
                    
                    if result["execution_failed"]:
                        execution_failures += 1
                    
                    evaluation_results.append({
                        "run_id": result["run_id"] + result["problem_idx"] * self.num_runs,  # Unique run ID
                        "is_correct": result["is_correct"],
                        "execution_time": result.get("execution_time", 0.0),
                        "problem_idx": result["problem_idx"],
                        "run_idx": result["run_id"]
                    })
                    
                    # Update progress
                    progress_printer.update(
                        increment=1,
                        fallback_triggered=result["execution_failed"],
                        timeout_triggered=False,
                        cancelled=False
                    )
                    
                except Exception as e:
                    print(f"Evaluation task failed: {e}")
                    execution_failures += 1
                    
                    # Create error result
                    evaluation_results.append({
                        "run_id": len(evaluation_results),  # Fallback run ID
                        "is_correct": False,
                        "execution_time": 0.0,
                        "problem_idx": -1,
                        "run_idx": -1
                    })
                    
                    progress_printer.update(
                        increment=1,
                        fallback_triggered=True,
                        timeout_triggered=False,
                        cancelled=False
                    )
        
        # Finish progress tracking
        progress_printer.close()
        
        # IMPORTANT: Get cost breakdown from aggregated costs (includes worker threads)
        total_eval_cost, evaluation_cost_breakdown = token_tracker.get_aggregated_costs()
        
        # Calculate per-run accuracies for standard error and standard deviation
        run_accuracies = []
        for run_idx in range(self.num_runs):
            run_results = [r for r in evaluation_results if r.get("run_idx") == run_idx]
            if run_results:
                correct_count = sum(1 for r in run_results if r["is_correct"])
                run_accuracy = correct_count / len(run_results)
                run_accuracies.append(run_accuracy)
        
        # Calculate statistics with both std_error and std_dev
        if run_accuracies:
            mean_accuracy = np.mean(run_accuracies)
            if len(run_accuracies) > 1:
                std_dev = np.std(run_accuracies, ddof=1)  # Sample standard deviation
                std_error = std_dev / np.sqrt(len(run_accuracies))  # Standard error
            else:
                std_dev = 0.0
                std_error = 0.0
        else:
            mean_accuracy = 0.0
            std_error = 0.0
            std_dev = 0.0
        
        # Calculate execution reliability
        total_evaluations = len(self.problems) * self.num_runs
        execution_reliability = (total_evaluations - execution_failures) / total_evaluations
        
        # Final performance score
        performance_score = mean_accuracy * execution_reliability
        
        # Full results dictionary with both metrics
        full_results = {
            "mean_accuracy": mean_accuracy,
            "std_error": std_error,
            "std_dev": std_dev,  # Add std_dev to results
            "performance_score": performance_score,
            "execution_reliability": execution_reliability,
            "run_accuracies": run_accuracies,
            "num_runs": self.num_runs,
            "num_problems": len(self.problems),
            "total_evaluations": total_evaluations,
            "execution_failures": execution_failures
        }
        
        # Cost info dictionary
        cost_info = {
            "total_cost": total_eval_cost,
            "from_cache": False,
            "model_breakdown": evaluation_cost_breakdown
        }
        
        # Update in-memory cache with all three metrics
        self.evaluation_cache[memory_cache_key] = (performance_score, std_error, std_dev, full_results)
        
        # Save to external cache
        if self.use_external_cache and self.external_cache and component_sequence:
            cache_saved = self.external_cache.put(
                dataset_type=self.dataset_type,
                is_test=self.is_test,
                num_problems=len(self.problems),
                num_runs=self.num_runs,
                model=BACKBONE_MODEL,
                judge_model=JUDGE_MODEL,
                component_sequence=component_sequence,
                performance_score=performance_score,
                std_error=std_error,
                full_results=full_results,
                system_code=system.get("code", ""),
                system_name=system_name
            )
            
            if not cache_saved:
                print(f"Warning: Failed to save {system_name} to external cache")
        
        # Show worker token usage after system evaluation
        worker_usage_tracker.print_stage_usage_summary(
            f"System: {system_name}", 
            [BACKBONE_MODEL, AGENT_MODEL, JUDGE_MODEL]
        )
        
        # Print performance with both std_error and std_dev
        print(f"System {system_name}: {mean_accuracy:.3f} ± {std_error:.3f} (se) ± {std_dev:.3f} (sd) accuracy, {execution_reliability:.3f} reliability = {performance_score:.3f} score")
        
        # Display cost breakdown
        if total_eval_cost > 0:
            print(f"\n💰 Cost for {system_name}: {format_cost(total_eval_cost)}")
            for model, model_data in evaluation_cost_breakdown.items():
                if model_data["total_cost"] > 0:
                    print(f"   {model}: {format_cost(model_data['total_cost'])} "
                          f"(Input: {format_cost(model_data['input_cost'])} / {model_data['input_tokens']:,} tokens, "
                          f"Output: {format_cost(model_data['output_cost'])} / {model_data['output_tokens']:,} tokens)")
        else:
            print(f"\n💰 Cost for {system_name}: {format_cost(0.0, is_cached=True)}")
        
        return performance_score, std_error, std_dev, full_results, cost_info
    
    def set_test_mode(self, is_test: bool = True):
        """Set whether evaluating test set (affects cache key)."""
        self.is_test = is_test
        if is_test:
            print("Evaluator set to TEST mode")
        else:
            print("Evaluator set to VALIDATION mode")
    
    def get_problem_set_info(self) -> Dict[str, Any]:
        """Get information about the current problem set."""
        if not self.problems:
            return {"error": "No problems loaded"}
        
        return {
            "dataset_type": self.dataset_type,
            "num_problems": len(self.problems),
            "num_runs": self.num_runs,
            "is_test": self.is_test,
            "total_evaluations_per_system": len(self.problems) * self.num_runs,
            "sample_problems": self.problems[:3] if len(self.problems) >= 3 else self.problems,
            "external_cache_enabled": self.use_external_cache,
            "method_version": self.method_version if self.use_external_cache else None
        }
    
    def clear_evaluation_cache(self):
        """Clear the in-memory evaluation cache (forces re-evaluation or cache lookup)."""
        self.evaluation_cache.clear()
        print("In-memory evaluation cache cleared")
    
    def print_cache_stats(self):
        """Print external cache statistics if enabled."""
        if self.use_external_cache and self.external_cache:
            self.external_cache.print_stats()
        else:
            print("External cache not enabled")
