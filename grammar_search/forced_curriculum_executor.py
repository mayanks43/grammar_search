#!/usr/bin/env python3
"""
Simplified Forced Component Curriculum Executor with strict n-ordering.
Two phases: Forced exploration with observation count ordering, then free Thompson/Random sampling.
"""

import json
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from grammar_search.forced_curriculum_sampler import SimplifiedForcedCurriculumSampler
from grammar_search.fixed_problem_evaluator import FixedProblemSetEvaluator
from grammar_search.grammar_rules import MODULAR_GRAMMAR_RULES, COMPONENT_TERMINALS
from grammar_search.weighted_grammar_config import PRODUCTION_WEIGHTS
from common.rolling_queue_processor import SimpleProgressPrinter
from common.token_tracker import token_tracker, format_cost

# Import for test evaluation
from common.data_utils import get_test_examples
from common.execution_utils import ThreadSafeAgentExecutor, SystemValidator
from common.answer_equivalence import are_answers_equivalent
from common.llm_interface import calculate_max_workers_for_model
from common.config import BACKBONE_MODEL, AGENT_MODEL, JUDGE_MODEL
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.worker_usage_tracker import worker_usage_tracker
from common.performance_stats import select_systems_for_test_grammar


class SimplifiedForcedCurriculumExecutor:
    """Simplified executor for forced curriculum with strict observation count ordering."""
    
    def __init__(self,
                 dataset_type: str = "gpqa",
                 total_iterations: int = 50,
                 forced_iterations: int = 30,
                 forced_max_length: int = 6,
                 free_max_length: int = 8,
                 # Sampling configuration
                 forced_sampling: str = "thompson",
                 forced_use_weights: bool = True,
                 free_sampling: str = "thompson",
                 free_use_weights: bool = True,
                 # Thompson parameters
                 credit_assignment: str = "bottleneck",
                 alpha_init: float = 1.0,
                 beta_init: float = 1.0,
                 # Evaluation configuration
                 num_problems: int = 32,
                 num_eval_runs: int = 4,
                 run_test_evaluation: bool = True,
                 test_eval_runs: int = 4,
                 num_top_systems: int = 1,
                 # Beta scores display
                 print_beta_scores: bool = True,
                 beta_scores_top_n: int = 10,
                 # External cache configuration
                 use_external_cache: bool = True,
                 cache_dir: str = None,
                 method_version: str = "v1"):
        """
        Initialize simplified forced curriculum executor.
        
        Args:
            dataset_type: Dataset for evaluation
            total_iterations: Total iteration budget
            forced_iterations: Number of iterations for forced exploration
            forced_max_length: Max sequence length during forced phase
            free_max_length: Max sequence length during free phase
            forced_sampling: Sampling method for forced phase
            forced_use_weights: Whether to use weights in forced phase
            free_sampling: Sampling method for free phase
            free_use_weights: Whether to use weights in free phase
            credit_assignment: Credit assignment strategy
            alpha_init: Initial alpha for Beta prior
            beta_init: Initial beta for Beta prior
            num_problems: Number of validation problems
            num_eval_runs: Runs per validation problem
            run_test_evaluation: Whether to run test evaluation
            test_eval_runs: Runs per test problem
            num_top_systems: Number of top systems for test
            print_beta_scores: Whether to print Beta scores
            beta_scores_top_n: Number of top productions to show
            use_external_cache: Whether to use external cache
            cache_dir: Directory for external cache
            method_version: Method version for cache isolation
        """

        random.seed(42)
        np.random.seed(42)

        self.dataset_type = dataset_type
        self.total_iterations = total_iterations
        self.forced_iterations = forced_iterations
        self.forced_max_length = forced_max_length
        self.free_max_length = free_max_length
        
        # Validate iteration budget
        self.free_iterations = total_iterations - forced_iterations
        if self.free_iterations < 0:
            raise ValueError(f"Forced iterations ({forced_iterations}) exceeds total ({total_iterations})")
        
        # Sampling configuration
        self.forced_sampling = forced_sampling
        self.forced_use_weights = forced_use_weights
        self.free_sampling = free_sampling
        self.free_use_weights = free_use_weights
        
        # Credit assignment
        self.credit_assignment = credit_assignment
        
        # Test configuration
        self.run_test_evaluation = run_test_evaluation
        self.test_eval_runs = test_eval_runs
        self.num_top_systems = num_top_systems
        
        # Beta scores printing
        self.print_beta_scores = print_beta_scores
        self.beta_scores_top_n = beta_scores_top_n
        
        # External cache configuration
        self.use_external_cache = use_external_cache
        self.cache_dir = cache_dir
        self.method_version = method_version
        
        # Initialize cost tracking
        self.phase_costs = {}
        self.cumulative_cost = 0.0
        self.system_costs = []
        self.cached_count = 0
        self.computed_count = 0
        
        # Clear any previous costs at initialization
        token_tracker.clear_aggregated_costs()
        
        # Initialize evaluator with external cache
        self.evaluator = FixedProblemSetEvaluator(
            dataset_type=dataset_type,
            num_problems=num_problems,
            num_runs=num_eval_runs,
            cache_dir=cache_dir,
            method_version=method_version,
            use_external_cache=use_external_cache
        )
        
        # Initialize sampler
        weights = PRODUCTION_WEIGHTS if (forced_use_weights or free_use_weights) else {}
        self.sampler = SimplifiedForcedCurriculumSampler(
            MODULAR_GRAMMAR_RULES,
            production_weights=weights,
            alpha_init=alpha_init,
            beta_init=beta_init,
            dataset_type=dataset_type,
        )
        
        # Results tracking
        self.results = []
        self.phase_results = defaultdict(list)
        self.best_system = None
        self.best_reward = -1
        self.all_systems = []
        self.seen_sequences = set()
        
        # Metadata
        self.metadata = {
            "dataset_type": dataset_type,
            "total_iterations": total_iterations,
            "forced_iterations": forced_iterations,
            "free_iterations": self.free_iterations,
            "forced_max_length": forced_max_length,
            "free_max_length": free_max_length,
            "forced_sampling": forced_sampling,
            "forced_use_weights": forced_use_weights,
            "free_sampling": free_sampling,
            "free_use_weights": free_use_weights,
            "credit_assignment": credit_assignment,
            "alpha_init": alpha_init,
            "beta_init": beta_init,
            "num_problems": num_problems,
            "num_eval_runs": num_eval_runs,
            "test_eval_runs": test_eval_runs,
            "num_top_systems": num_top_systems,
            "print_beta_scores": print_beta_scores,
            "beta_scores_top_n": beta_scores_top_n,
            "use_external_cache": use_external_cache,
            "cache_dir": cache_dir,
            "method_version": method_version,
            "cost_tracking_enabled": True,
            "std_dev_reporting_enabled": True
        }
    
    def _print_schedule(self):
        """Print the simplified experiment schedule."""
        print("\n" + "=" * 60)
        print("SIMPLIFIED FORCED CURRICULUM SCHEDULE")
        print("=" * 60)
        
        print(f"\nDataset: {self.dataset_type}")
        print(f"Total iterations: {self.total_iterations}")
        
        print(f"\nForced Exploration Phase:")
        print(f"  Iterations: {self.forced_iterations}")
        print(f"  Max length: {self.forced_max_length}")
        print(f"  Sampling: {self.forced_sampling}")
        print(f"  Weights: {'enabled' if self.forced_use_weights else 'disabled'}")
        print(f"  Strategy: Strict n-ordering (exhaust n=0, then n=1, etc.)")
        
        print(f"\nFree Exploration Phase:")
        print(f"  Iterations: {self.free_iterations}")
        print(f"  Max length: {self.free_max_length}")
        print(f"  Sampling: {self.free_sampling}")
        print(f"  Weights: {'enabled' if self.free_use_weights else 'disabled'}")
        print(f"  Strategy: Pure Thompson sampling")
        
        print(f"\nCredit Assignment: {self.credit_assignment}")
        print(f"Thompson Sampling: α={self.sampler.alpha_init}, β={self.sampler.beta_init}")
        print(f"Beta Scores: {'enabled' if self.print_beta_scores else 'disabled'} (top {self.beta_scores_top_n})")
        
        print(f"\nEvaluation Configuration:")
        print(f"  Validation: {self.evaluator.num_problems} problems × {self.evaluator.num_runs} runs")
        if self.run_test_evaluation:
            print(f"  Test: top {self.num_top_systems} system(s) × {self.test_eval_runs} runs")
        else:
            print(f"  Test: DISABLED")
        
        print(f"\nCache Configuration:")
        print(f"  External cache: {'ENABLED' if self.use_external_cache else 'DISABLED'}")
        if self.use_external_cache:
            print(f"  Cache directory: {self.cache_dir or '~/.fusion_adas_cache'}")
            print(f"  Method version: {self.method_version}")
        
        print("=" * 60)
        print()
    
    def _print_beta_scores(self, iteration: int, top_n: int = None):
        """Print Beta distribution scores for top production rules."""
        if not self.print_beta_scores:
            return
        
        if top_n is None:
            top_n = self.beta_scores_top_n
        
        production_stats = self.sampler.production_stats
        
        if not production_stats:
            return
        
        # Collect and sort production rules by mean Beta score
        rule_scores = []
        for (non_terminal, production), stats in production_stats.items():
            mean_score = stats.get_mean()
            rule_scores.append({
                "non_terminal": non_terminal,
                "production": production,
                "mean": mean_score,
                "alpha": stats.alpha,
                "beta": stats.beta,
                "observations": stats.n,
                "confidence": 1.0 / (1.0 + np.sqrt(stats.n)) if stats.n > 0 else 0.0
            })
        
        # Sort by mean score descending
        rule_scores.sort(key=lambda x: x["mean"], reverse=True)
        
        # Print compact Beta scores
        print(f"\n    [Beta Scores - Iter {iteration}] Top {min(top_n, len(rule_scores))} productions:")
        for i, rule in enumerate(rule_scores[:top_n]):
            nt = rule["non_terminal"]
            prod = rule["production"]
            
            print(f"      {i+1:2}. {nt} → {prod}")
            print(f"          Mean: {rule['mean']:.3f} | α={rule['alpha']:.1f}, β={rule['beta']:.1f} | "
                  f"n={rule['observations']} | conf={rule['confidence']:.2f}")
    
    def _print_phase_cost_summary(self, phase_name: str, phase_cost: float, phase_breakdown: Dict):
        """Print cost summary for a completed phase."""
        print(f"\n--- {phase_name} Cost Summary ---")
        print(f"Systems evaluated: {self.computed_count + self.cached_count} total ({self.computed_count} computed, {self.cached_count} cached)")
        print(f"Phase cost: {format_cost(phase_cost)}")
        
        if phase_breakdown:
            print(f"  By model:")
            for model, data in phase_breakdown.items():
                if data["total_cost"] > 0:
                    print(f"    {model}: {format_cost(data['total_cost'])} "
                          f"({data['input_tokens']:,} input / {data['output_tokens']:,} output tokens)")
        
        print(f"Cumulative experiment cost: {format_cost(self.cumulative_cost)}")
    
    def run_forced_exploration_phase(self) -> int:
        """
        Run forced exploration with strict observation count ordering.
        
        Returns:
            Number of iterations executed
        """
        print(f"\n--- Forced Exploration Phase ({self.forced_iterations} iterations, max_length={self.forced_max_length}) ---")
        
        # Track phase start
        phase_start_cached = self.cached_count
        phase_start_computed = self.computed_count
        phase_start_cost = self.cumulative_cost
        
        # Configure sampling for forced phase
        self.sampler.set_sampling_mode(self.forced_sampling, self.forced_use_weights)
        
        iterations_used = 0
        current_n_threshold = 0
        
        while iterations_used < self.forced_iterations:
            # Get terminals at current n threshold - REFRESH EACH TIME
            terminals_at_n = self.sampler.get_terminals_at_threshold(current_n_threshold)
            
            if not terminals_at_n:
                # No terminals at this level, move to next
                current_n_threshold += 1
                # Check if we've cycled through all possible n values
                max_n = 0
                for terminal in COMPONENT_TERMINALS:
                    terminal_n = self.sampler.get_terminal_observation_count(terminal)
                    if terminal_n > max_n:
                        max_n = terminal_n
                
                if current_n_threshold > max_n:
                    current_n_threshold = 0  # Wrap back to lowest
                    print(f"  Wrapping back to n=0 (max observed n={max_n})")
                continue
            
            # Process ONE terminal at a time, then re-check the threshold
            terminal = terminals_at_n[0]  # Take the first terminal
            
            # Print stats before forcing this specific terminal
            print(f"\n  Current n threshold: {current_n_threshold}")
            print(f"  Terminals at n={current_n_threshold}: {len(terminals_at_n)}")
            for t in terminals_at_n:  # Show ALL terminals
                print(f"    - {t}")
            
            # Get full distribution periodically
            # if iterations_used % 10 == 0 or iterations_used == 0:
            all_terminal_obs = {}
            for t in COMPONENT_TERMINALS:
                all_terminal_obs[t] = self.sampler.get_terminal_observation_count(t)
            
            obs_distribution = defaultdict(list)
            for t, n_obs in all_terminal_obs.items():
                obs_distribution[n_obs].append(t)
            
            print(f"\n  Full terminal observation distribution:")
            for n_val in sorted(obs_distribution.keys()):
                print(f"    n={n_val}: {len(obs_distribution[n_val])} terminals")
                for t in obs_distribution[n_val]:  # Show ALL terminals at each n
                    print(f"      - {t}")
        
            iteration = len(self.results) + 1
            terminal_n_before = self.sampler.get_terminal_observation_count(terminal)
            print(f"\n  Iter {iteration}: Forcing {terminal} (Beta n={terminal_n_before})")
            
            # Try to force this terminal
            sequence, derivation, attempts = self.sampler.force_component_with_max_length(
                terminal, self.forced_max_length
            )
            
            if not sequence:
                # Failed to force
                print(f"    Failed to force after {attempts} attempts")
                continue  # Don't increment iterations_used, try next terminal
            
            # Check for duplicates
            sequence_key = tuple(sequence)
            is_duplicate = sequence_key in self.seen_sequences
            
            if is_duplicate:
                print(f"    [Duplicate] {sequence} (len {len(sequence)})")
                continue  # Skip without updating Beta stats or counting as iteration
            
            # New unique sequence
            print(f"    Found: {sequence} (len {len(sequence)}, {attempts} attempts)")
            self.seen_sequences.add(sequence_key)
            
            # Update component coverage
            self.sampler.update_component_coverage("forced_exploration", sequence)
            
            # Generate and evaluate system
            system = self.sampler.get_or_generate_system(sequence)
            reward, std_error, std_dev, full_results, cost_info = self.evaluator.evaluate_system(system)
            
            # Track costs
            if cost_info.get("from_cache", False):
                self.cached_count += 1
            else:
                self.computed_count += 1
                self.cumulative_cost += cost_info.get("total_cost", 0.0)
            
            # Store system cost
            self.system_costs.append({
                "system_name": system["name"],
                "iteration": iteration,
                "phase": "forced_exploration",
                "cost": cost_info.get("total_cost", 0.0),
                "from_cache": cost_info.get("from_cache", False)
            })
            
            # Update Thompson statistics
            self.sampler.update_stats(derivation, reward, self.credit_assignment)
            
            # Check what n values changed after update
            terminal_n_after = self.sampler.get_terminal_observation_count(terminal)
            print(f"    Terminal {terminal} Beta n: {terminal_n_before} → {terminal_n_after}")
            
            # Also check other terminals in the sequence
            for component in sequence:
                if component != terminal and component in COMPONENT_TERMINALS:
                    component_n = self.sampler.get_terminal_observation_count(component)
                    print(f"    Also observed: {component} (now Beta n={component_n})")
            
            # Store system
            system_record = system.copy()
            system_record['reward'] = reward
            system_record['std_error'] = std_error
            system_record['std_dev'] = std_dev
            system_record['full_results'] = full_results
            system_record['iteration'] = iteration
            system_record['phase'] = "forced_exploration"
            system_record['target_terminal'] = terminal
            system_record['n_threshold'] = current_n_threshold
            system_record['force_attempts'] = attempts
            system_record['is_duplicate'] = False
            system_record['evaluation_cost'] = cost_info.get("total_cost", 0.0)
            system_record['from_cache'] = cost_info.get("from_cache", False)
            self.all_systems.append(system_record)
            
            # Track best system
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_system = system_record.copy()
                self.best_system['discovered_at_iteration'] = iteration
                self.best_system['discovered_in_phase'] = "forced_exploration"
            
            # Store results
            self.results.append({
                "iteration": iteration,
                "phase": "forced_exploration",
                "target_terminal": terminal,
                "n_threshold": current_n_threshold,
                "actual_length": len(sequence),
                "components": sequence,
                "reward": reward,
                "std_error": std_error,
                "std_dev": std_dev,
                "system_name": system["name"],
                "force_attempts": attempts,
                "is_duplicate": False,
                "evaluation_cost": cost_info.get("total_cost", 0.0),
                "from_cache": cost_info.get("from_cache", False)
            })
            self.phase_results["forced_exploration"].append(reward)
            
            # Print Beta scores
            self._print_beta_scores(iteration)
            
            iterations_used += 1
        
        # Update sampler's current n threshold
        self.sampler.current_n_threshold = current_n_threshold
        
        # Calculate phase costs
        phase_cost = self.cumulative_cost - phase_start_cost
        total_cost, model_breakdown = token_tracker.get_aggregated_costs()
        
        # Store phase costs
        self.phase_costs["forced_exploration"] = {
            "phase_cost": phase_cost,
            "model_breakdown": model_breakdown,
            "computed": self.computed_count - phase_start_computed,
            "cached": self.cached_count - phase_start_cached
        }
        
        # Print phase summary
        self.sampler.print_phase_summary("forced_exploration")
        self._print_phase_cost_summary("Forced Exploration", phase_cost, model_breakdown)
        
        return iterations_used

    def run_free_exploration_phase(self) -> int:
        """
        Run free exploration with pure Thompson sampling.
        
        Returns:
            Number of iterations executed
        """
        if self.free_iterations <= 0:
            print("\n--- Free Exploration Phase SKIPPED (0 iterations) ---")
            return 0
        
        print(f"\n--- Free Exploration Phase ({self.free_iterations} iterations, max_length={self.free_max_length}) ---")
        
        # Track phase start
        phase_start_cached = self.cached_count
        phase_start_computed = self.computed_count
        phase_start_cost = self.cumulative_cost
        
        # Configure sampling for free phase
        self.sampler.set_sampling_mode(self.free_sampling, self.free_use_weights)
        
        iterations_completed = 0
        
        while iterations_completed < self.free_iterations:
            iteration = len(self.results) + 1
            
            # Sample with max length constraint
            sequence, derivation = self.sampler.sample_with_max_length(self.free_max_length)
            
            # Check for duplicates
            sequence_key = tuple(sequence)
            is_duplicate = sequence_key in self.seen_sequences
            
            if is_duplicate:
                print(f"  Iter {iteration}: [Duplicate] {sequence} (len {len(sequence)})")
                continue  # Skip without updating Beta stats or counting as iteration
            
            # New unique sequence
            print(f"  Iter {iteration}: {sequence} (len {len(sequence)})")
            self.seen_sequences.add(sequence_key)
            
            # Update component coverage
            self.sampler.update_component_coverage("free_exploration", sequence)
            
            # Generate and evaluate system
            system = self.sampler.get_or_generate_system(sequence)
            reward, std_error, std_dev, full_results, cost_info = self.evaluator.evaluate_system(system)
            
            # Track costs
            if cost_info.get("from_cache", False):
                self.cached_count += 1
            else:
                self.computed_count += 1
                self.cumulative_cost += cost_info.get("total_cost", 0.0)
            
            # Store system cost
            self.system_costs.append({
                "system_name": system["name"],
                "iteration": iteration,
                "phase": "free_exploration",
                "cost": cost_info.get("total_cost", 0.0),
                "from_cache": cost_info.get("from_cache", False)
            })
            
            # Update Thompson statistics
            self.sampler.update_stats(derivation, reward, self.credit_assignment)
            
            # Store system
            system_record = system.copy()
            system_record['reward'] = reward
            system_record['std_error'] = std_error
            system_record['std_dev'] = std_dev
            system_record['full_results'] = full_results
            system_record['iteration'] = iteration
            system_record['phase'] = "free_exploration"
            system_record['is_duplicate'] = False
            system_record['evaluation_cost'] = cost_info.get("total_cost", 0.0)
            system_record['from_cache'] = cost_info.get("from_cache", False)
            self.all_systems.append(system_record)
            
            # Track best system
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_system = system_record.copy()
                self.best_system['discovered_at_iteration'] = iteration
                self.best_system['discovered_in_phase'] = "free_exploration"
            
            # Store results
            self.results.append({
                "iteration": iteration,
                "phase": "free_exploration",
                "actual_length": len(sequence),
                "components": sequence,
                "reward": reward,
                "std_error": std_error,
                "std_dev": std_dev,
                "system_name": system["name"],
                "is_duplicate": False,
                "evaluation_cost": cost_info.get("total_cost", 0.0),
                "from_cache": cost_info.get("from_cache", False)
            })
            self.phase_results["free_exploration"].append(reward)
            
            # Print Beta scores
            self._print_beta_scores(iteration)
            
            iterations_completed += 1
        
        # Calculate phase costs
        phase_cost = self.cumulative_cost - phase_start_cost
        total_cost, model_breakdown = token_tracker.get_aggregated_costs()
        
        # Store phase costs
        self.phase_costs["free_exploration"] = {
            "phase_cost": phase_cost,
            "model_breakdown": model_breakdown,
            "computed": self.computed_count - phase_start_computed,
            "cached": self.cached_count - phase_start_cached
        }
        
        # Print phase summary
        self.sampler.print_phase_summary("free_exploration")
        self._print_phase_cost_summary("Free Exploration", phase_cost, model_breakdown)
        
        return iterations_completed    

    def find_top_systems_within_stderr(self, all_systems: List[Dict], num_systems: int) -> List[Dict]:
        """Find top systems using statistical selection."""
        return select_systems_for_test_grammar(all_systems, num_systems, alpha=0.05)
    
    def _evaluate_system_on_test(self, system: Dict, test_problems: List[Dict]) -> Dict:
        """Evaluate a single system on the test set."""
        # Set evaluator to test mode
        self.evaluator.set_test_mode(True)
        
        # Create temporary test evaluator
        test_evaluator = FixedProblemSetEvaluator(
            dataset_type=self.dataset_type,
            num_problems=len(test_problems),
            num_runs=self.test_eval_runs,
            cache_dir=self.cache_dir,
            method_version=self.method_version,
            use_external_cache=self.use_external_cache
        )
        test_evaluator.set_test_mode(True)
        test_evaluator.problems = test_problems
        
        # Evaluate
        performance_score, std_error, std_dev, full_results, cost_info = test_evaluator.evaluate_system(system)
        
        # Reset to validation mode
        self.evaluator.set_test_mode(False)
        
        # Track test cost
        if cost_info.get("from_cache", False):
            self.cached_count += 1
        else:
            self.computed_count += 1
            self.cumulative_cost += cost_info.get("total_cost", 0.0)
        
        # Format results
        return {
            "dataset_type": self.dataset_type,
            "num_problems": len(test_problems),
            "num_runs": self.test_eval_runs,
            "total_evaluations": len(test_problems) * self.test_eval_runs,
            "successful_executions": int(full_results.get("execution_reliability", 0) * len(test_problems) * self.test_eval_runs),
            "execution_rate": full_results.get("execution_reliability", 0),
            "test_accuracy": full_results.get("mean_accuracy", 0),
            "test_std_error": std_error,
            "test_std_dev": std_dev,
            "run_accuracies": full_results.get("run_accuracies", []),
            "elapsed_time": 0.0,
            "system_name": system['name'],
            "validation_reward": system.get('reward', 0.0),
            "from_cache": cost_info.get("from_cache", False),
            "test_cost": cost_info.get("total_cost", 0.0)
        }
    
    def evaluate_top_systems_on_test(self, all_systems: List[Dict]) -> Dict[str, Any]:
        """Evaluate the top N systems on the test set."""
        # Find top systems
        top_systems = self.find_top_systems_within_stderr(all_systems, self.num_top_systems)
        
        if not top_systems:
            print("No systems to evaluate on test set")
            return None
        
        # Load test data
        print(f"\nLoading {self.dataset_type} test data...")
        test_problems = get_test_examples(self.dataset_type)
        
        print(f"Loaded {len(test_problems)} test problems")
        
        # Track test phase start
        test_start_cost = self.cumulative_cost
        
        # Evaluate each top system
        test_results = {}
        best_test_accuracy = 0.0
        best_test_system = None
        
        validator = SystemValidator(max_iterations=10)
        
        for idx, system in enumerate(top_systems, 1):
            print(f"\n--- Evaluating System {idx}/{len(top_systems)}: {system['name']} ---")
            print(f"  Validation reward: {system['reward']:.3f} ± {system.get('std_error', 0.0):.3f} (se) ± {system.get('std_dev', 0.0):.3f} (sd)")
            print(f"  Discovered at iteration: {system.get('iteration', 'N/A')}")
            print(f"  Phase: {system.get('phase', 'N/A')}")
            
            # Validate system
            is_safe, safety_error = validator.validate_system_code(system)
            if not is_safe:
                print(f"  System failed safety validation: {safety_error}")
                test_results[system['name']] = {"error": safety_error}
                continue
            
            # Evaluate on test set
            system_test_result = self._evaluate_system_on_test(system, test_problems)
            test_results[system['name']] = system_test_result
            
            # Show if result was from cache
            if system_test_result.get("from_cache"):
                print(f"  📦 Test results retrieved from cache")
            
            # Print test results
            print(f"\n  📊 Test Results: {system_test_result['test_accuracy']:.1%} ± {system_test_result['test_std_error']:.1%} (se) ± {system_test_result['test_std_dev']:.1%} (sd)")
            print(f"  💰 Test evaluation cost: {format_cost(system_test_result['test_cost'], system_test_result.get('from_cache', False))}")
            
            # Track best test system
            if system_test_result['test_accuracy'] > best_test_accuracy:
                best_test_accuracy = system_test_result['test_accuracy']
                best_test_system = system['name']
        
        # Calculate test phase cost
        test_total_cost = self.cumulative_cost - test_start_cost
        
        # Get test phase cost breakdown
        _, test_model_breakdown = token_tracker.get_aggregated_costs()
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        for sys_name, result in test_results.items():
            if 'error' not in result:
                cache_marker = " [cached]" if result.get("from_cache") else ""
                print(f"{sys_name}{cache_marker}:")
                print(f"  Test accuracy: {result['test_accuracy']:.1%} ± {result['test_std_error']:.1%} (se) ± {result['test_std_dev']:.1%} (sd)")
                print(f"  Execution rate: {result['execution_rate']:.1%}")
                print(f"  Test cost: {format_cost(result['test_cost'], result.get('from_cache', False))}")
        
        if best_test_system:
            print(f"\nBest test performance: {best_test_system} ({best_test_accuracy:.1%})")
        
        print(f"\nTotal test evaluation cost: {format_cost(test_total_cost)}")
        
        # Return format based on number of systems
        if self.num_top_systems == 1 and len(test_results) == 1:
            return list(test_results.values())[0]
        
        return {
            "num_systems_tested": len(top_systems),
            "systems_tested": [s['name'] for s in top_systems],
            "individual_results": test_results,
            "best_test_system": best_test_system,
            "best_test_accuracy": best_test_accuracy,
            "total_test_cost": test_total_cost,
            "test_cost_breakdown": test_model_breakdown
        }
    
    def _print_log_summary(self):
        """Print summary of debug log locations."""
        from common.debug_logger import debug_logger
        import os
        
        DEBUG_LOG_DIR = debug_logger.get_log_directory()
        
        print("\n" + "="*60)
        print("DEBUG LOG LOCATIONS")
        print("="*60)
        
        # Generation logs
        generation_dir = os.path.join(DEBUG_LOG_DIR, "generation")
        if os.path.exists(generation_dir):
            generated_systems = [d for d in os.listdir(generation_dir) if os.path.isdir(os.path.join(generation_dir, d))]
            print(f"\nGeneration logs ({len(generated_systems)} systems):")
            print(f"  {generation_dir}")
        
        # Evaluation logs
        evaluation_dir = os.path.join(DEBUG_LOG_DIR, "evaluation")
        if os.path.exists(evaluation_dir):
            eval_systems = [d for d in os.listdir(evaluation_dir) 
                          if os.path.isdir(os.path.join(evaluation_dir, d)) and not d.startswith("test_")]
            print(f"\nValidation evaluation logs ({len(eval_systems)} systems):")
            print(f"  {evaluation_dir}")
        
        print("="*60)
    
    def print_cache_summary(self):
        """Print external cache statistics."""
        if self.use_external_cache and self.evaluator.external_cache:
            print("\n" + "="*60)
            print("EXTERNAL CACHE SUMMARY")
            print("="*60)
            
            stats = self.evaluator.external_cache.get_stats()
            
            print(f"Method version: {self.method_version}")
            print(f"Total cache entries: {stats['total_entries']}")
            print(f"Total cache size: {stats['total_size_mb']:.2f} MB")
            
            # Count entries for current method version
            current_version_count = stats['by_method_version'].get(self.method_version, 0)
            print(f"Entries for this version: {current_version_count}")
            
            # Count entries for current dataset
            current_dataset_count = stats['by_dataset'].get(self.dataset_type, 0)
            print(f"Entries for {self.dataset_type}: {current_dataset_count}")
            
            print("="*60)
    
    def print_cost_summary(self):
        """Print comprehensive cost summary."""
        final_total_cost, model_breakdown = token_tracker.get_aggregated_costs()
        
        print("\n" + "="*60)
        print("EXPERIMENT COST SUMMARY")
        print("="*60)
        print(f"Total cost: {format_cost(self.cumulative_cost)}")
        
        # Cost by phase
        if self.phase_costs:
            print(f"\nCost by phase:")
            for phase_name, phase_data in self.phase_costs.items():
                phase_cost = phase_data.get("phase_cost", 0.0)
                phase_computed = phase_data.get("computed", 0)
                phase_cached = phase_data.get("cached", 0)
                print(f"  {phase_name}: {format_cost(phase_cost)} ({phase_computed} computed, {phase_cached} cached)")
        
        # Model breakdown
        if model_breakdown:
            print(f"\nCost by model:")
            for model, data in model_breakdown.items():
                if data["total_cost"] > 0:
                    print(f"  {model}: {format_cost(data['total_cost'])}")
                    print(f"    Input: {format_cost(data['input_cost'])} ({data['input_tokens']:,} tokens)")
                    print(f"    Output: {format_cost(data['output_cost'])} ({data['output_tokens']:,} tokens)")
                    print(f"    Total calls: {data['calls']:,}")
        
        # Cache statistics
        print(f"\nEvaluation statistics:")
        print(f"  Cached evaluations: {self.cached_count}")
        print(f"  Computed evaluations: {self.computed_count}")
        print(f"  Cache hit rate: {(self.cached_count / (self.cached_count + self.computed_count) * 100):.1f}%" 
              if (self.cached_count + self.computed_count) > 0 else "0.0%")
        
        # Average cost per system
        if self.computed_count > 0:
            avg_cost = self.cumulative_cost / self.computed_count
            print(f"  Average cost per computed system: {format_cost(avg_cost)}")
        
        print("="*60)
    
    def run(self) -> Dict[str, Any]:
        """Run the simplified forced curriculum experiment."""
        if not self.evaluator.initialize_problem_set():
            raise RuntimeError("Failed to initialize problem set")
        
        print("Simplified Forced Component Curriculum Thompson Sampling")
        print("=" * 60)
        print(f"Dataset: {self.dataset_type}, Iterations: {self.total_iterations}")
        print(f"Components: {len(COMPONENT_TERMINALS)}")
        
        # Print schedule
        self._print_schedule()
        
        self.metadata["start_time"] = time.time()
        
        progress = SimpleProgressPrinter(
            total_tasks=self.total_iterations,
            log_interval=max(5, self.total_iterations // 10)
        )
        
        # Forced exploration phase
        forced_iters = self.run_forced_exploration_phase()
        for _ in range(forced_iters):
            progress.update(1)
        
        # Free exploration phase
        free_iters = self.run_free_exploration_phase()
        for _ in range(free_iters):
            progress.update(1)
        
        progress.close()
        
        self.metadata["end_time"] = time.time()
        self.metadata["total_time"] = self.metadata["end_time"] - self.metadata["start_time"]
        
        # Print log summary
        self._print_log_summary()
        
        # Test evaluation
        test_results = None
        if self.run_test_evaluation and self.all_systems:
            print("\n" + "="*60)
            print("EVALUATING TOP SYSTEMS ON TEST SET")
            print("="*60)
            test_results = self.evaluate_top_systems_on_test(self.all_systems)
        
        # Print cost summary
        self.print_cost_summary()
        
        return self.get_results(test_results)
    
    def get_results(self, test_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Get complete results with analysis."""
        analysis = self.analyze_results()
        
        # Get final cost breakdown
        final_cumulative_cost, model_breakdown = token_tracker.get_aggregated_costs()
        
        results = {
            "metadata": self.metadata,
            "results": self.results,
            "phase_summary": self.phase_results,
            "analysis": analysis,
            "sampler_stats": self.sampler.get_stats_summary(),
            "component_coverage": self.sampler.get_component_coverage_stats(),
            "best_system": self.best_system,
            "all_systems": self.all_systems,
            "duplicate_rate": len([r for r in self.results if r.get('is_duplicate', False)]) / len(self.results) if self.results else 0,
            "unique_sequences": len(self.seen_sequences),
            "cost_tracking": {
                "total_cost": self.cumulative_cost,
                "phase_costs": self.phase_costs,
                "system_costs": self.system_costs,
                "model_breakdown": model_breakdown,
                "cached_evaluations": self.cached_count,
                "computed_evaluations": self.computed_count,
                "cache_hit_rate": (self.cached_count / (self.cached_count + self.computed_count)) if (self.cached_count + self.computed_count) > 0 else 0
            }
        }
        
        if test_results:
            results["test_evaluation"] = test_results
        
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        if not self.results:
            return {}
        
        all_rewards = [r["reward"] for r in self.results]
        all_std_errors = [r.get("std_error", 0.0) for r in self.results if "std_error" in r]
        all_std_devs = [r.get("std_dev", 0.0) for r in self.results if "std_dev" in r]
        
        analysis = {
            "overall": {
                "mean_reward": np.mean(all_rewards),
                "std_reward": np.std(all_rewards),
                "max_reward": np.max(all_rewards),
                "min_reward": np.min(all_rewards),
                "final_10_mean": np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards),
                "mean_std_error": np.mean(all_std_errors) if all_std_errors else 0.0,
                "mean_std_dev": np.mean(all_std_devs) if all_std_devs else 0.0
            },
            "by_phase": {}
        }
        
        # Analyze each phase
        for phase, rewards in self.phase_results.items():
            if rewards:
                phase_results = [r for r in self.results if r.get("phase") == phase]
                phase_std_errors = [r.get("std_error", 0.0) for r in phase_results if "std_error" in r]
                phase_std_devs = [r.get("std_dev", 0.0) for r in phase_results if "std_dev" in r]
                
                analysis["by_phase"][phase] = {
                    "count": len(rewards),
                    "mean": np.mean(rewards),
                    "std": np.std(rewards),
                    "max": np.max(rewards),
                    "min": np.min(rewards),
                    "mean_std_error": np.mean(phase_std_errors) if phase_std_errors else 0.0,
                    "mean_std_dev": np.mean(phase_std_devs) if phase_std_devs else 0.0
                }
        
        # Learning progression
        if len(all_rewards) >= 20:
            first_quarter = all_rewards[:len(all_rewards)//4]
            last_quarter = all_rewards[3*len(all_rewards)//4:]
            
            analysis["progression"] = {
                "first_quarter_mean": np.mean(first_quarter),
                "last_quarter_mean": np.mean(last_quarter),
                "improvement": np.mean(last_quarter) - np.mean(first_quarter)
            }
        
        # Best system info
        if self.best_system:
            analysis["best_system"] = {
                "name": self.best_system['name'],
                "reward": self.best_reward,
                "std_error": self.best_system.get('std_error', 0.0),
                "std_dev": self.best_system.get('std_dev', 0.0),
                "discovered_at_iteration": self.best_system.get('discovered_at_iteration'),
                "phase": self.best_system.get('discovered_in_phase'),
                "components": self.best_system.get('component_sequence', []),
                "evaluation_cost": self.best_system.get('evaluation_cost', 0.0)
            }
        
        # Forcing efficiency (for forced phase)
        force_attempts = [r.get('force_attempts', 0) for r in self.results 
                          if r.get('phase') == 'forced_exploration' and 'force_attempts' in r and r.get('force_attempts', 0) > 0]
        if force_attempts:
            analysis["forcing_efficiency"] = {
                "mean_attempts": np.mean(force_attempts),
                "median_attempts": np.median(force_attempts),
                "max_attempts": max(force_attempts),
                "min_attempts": min(force_attempts)
            }
        
        # Terminal observation analysis
        if self.sampler:
            terminal_obs = {}
            for terminal in COMPONENT_TERMINALS:
                terminal_obs[terminal] = self.sampler.get_terminal_observation_count(terminal)
            
            obs_values = list(terminal_obs.values())
            if obs_values:
                analysis["terminal_observations"] = {
                    "min_n": min(obs_values),
                    "max_n": max(obs_values),
                    "mean_n": np.mean(obs_values),
                    "median_n": np.median(obs_values),
                    "terminals_at_n0": sum(1 for v in obs_values if v == 0),
                    "terminals_explored": sum(1 for v in obs_values if v > 0)
                }
        
        # Duplicate analysis
        duplicates = [r for r in self.results if r.get('is_duplicate', False)]
        if duplicates:
            analysis["duplicate_analysis"] = {
                "total_duplicates": len(duplicates),
                "duplicate_rate": len(duplicates) / len(self.results),
                "duplicates_by_phase": {}
            }
            for phase in set(r['phase'] for r in self.results):
                phase_results = [r for r in self.results if r['phase'] == phase]
                phase_duplicates = [r for r in phase_results if r.get('is_duplicate', False)]
                analysis["duplicate_analysis"]["duplicates_by_phase"][phase] = {
                    "count": len(phase_duplicates),
                    "rate": len(phase_duplicates) / len(phase_results) if phase_results else 0
                }
        
        # Cost analysis
        if self.system_costs:
            computed_costs = [s['cost'] for s in self.system_costs if not s['from_cache']]
            if computed_costs:
                analysis["cost_analysis"] = {
                    "mean_cost_per_system": np.mean(computed_costs),
                    "median_cost_per_system": np.median(computed_costs),
                    "max_cost_per_system": max(computed_costs),
                    "min_cost_per_system": min(computed_costs),
                    "total_computed_cost": sum(computed_costs),
                    "systems_from_cache": len([s for s in self.system_costs if s['from_cache']]),
                    "systems_computed": len(computed_costs)
                }
        
        return analysis
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"simplified_forced_curriculum_{self.dataset_type}_{timestamp}.json"
        
        # Create directory if needed
        import os
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        results = self.get_results()
        
        # Make JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            return obj
        
        with open(filename, 'w') as f:
            json.dump(make_serializable(results), f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename
