#!/usr/bin/env python3
"""
Simplified main script to run forced component curriculum Grammar Search experiments.
Two phases: Forced exploration with strict n-ordering, then free Thompson/Random sampling.
"""

import argparse
import sys
import os
import json
import random
import numpy as np
from datetime import datetime
from grammar_search.forced_curriculum_executor import SimplifiedForcedCurriculumExecutor
from grammar_search.grammar_rules import COMPONENT_TERMINALS
from common.config import USE_OPENAI


class TeeOutput:
    """Context manager to duplicate stdout/stderr to both console and file."""
    
    def __init__(self, filename):
        self.file = open(filename, 'w', buffering=1)  # Line buffering
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()


def validate_iteration_budget(args):
    """Validate iteration budget split."""
    if args.forced_iterations > args.iterations:
        print(f"ERROR: Forced iterations ({args.forced_iterations}) exceeds total iterations ({args.iterations})")
        sys.exit(1)
    
    free_iterations = args.iterations - args.forced_iterations
    print(f"Budget allocation: {args.forced_iterations} forced + {free_iterations} free = {args.iterations} total")
    return args.forced_iterations, free_iterations


def setup_logging(args):
    """Set up console output logging to file."""
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_suffix = f"_{args.method_version}" if not args.no_cache else ""
        log_filename = f"simplified_forced_curriculum_{args.dataset}{cache_suffix}_{timestamp}.log"
    else:
        log_filename = args.log_file
    
    # Create log directory if needed
    log_dir = os.path.dirname(log_filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    print(f"Logging output to: {log_filename}")
    return log_filename


def setup_debug_directory(args):
    """Set up debug log directory."""
    from common.debug_logger import debug_logger
    
    if args.debug_dir:
        debug_logger.set_log_directory(args.debug_dir)
        print(f"Debug logs directory: {args.debug_dir}")
    else:
        # Set default debug directory based on dataset
        default_dir = f"simplified_forced_{args.dataset}_debug_logs"
        debug_logger.set_log_directory(default_dir)
        print(f"Debug logs directory: {default_dir}")


def print_experiment_header(args, num_components):
    """Print experiment header information."""
    print(f"Log started at: {datetime.now().isoformat()}")
    print(f"Command: python {' '.join(sys.argv)}")
    print("="*60)
    print(f"Using {'OpenAI' if USE_OPENAI else 'Azure'} backend")
    print(f"Found {num_components} components in grammar")
    print(f"Dataset: {args.dataset}")
    print(f"Total iterations: {args.iterations}")
    print(f"Forced iterations: {args.forced_iterations}")
    print(f"Free iterations: {args.iterations - args.forced_iterations}")
    print(f"External cache: {'ENABLED' if not args.no_cache else 'DISABLED'}")
    if not args.no_cache:
        print(f"Cache directory: {args.cache_dir or '~/.fusion_adas_cache'}")
        print(f"Method version: {args.method_version}")
    print("="*60)


def print_experiment_summary(results, args):
    """Print comprehensive experiment summary."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    analysis = results.get("analysis", {})
    
    # Overall performance
    if "overall" in analysis:
        overall = analysis["overall"]
        print(f"\nOverall Performance:")
        print(f"  Mean reward: {overall['mean_reward']:.3f}")
        print(f"  Std deviation: {overall['std_reward']:.3f}")
        print(f"  Best reward: {overall['max_reward']:.3f}")
        print(f"  Worst reward: {overall['min_reward']:.3f}")
        print(f"  Final 10 mean: {overall['final_10_mean']:.3f}")
    
    # Learning progression
    if "progression" in analysis:
        prog = analysis["progression"]
        print(f"\nLearning Progression:")
        print(f"  First quarter: {prog['first_quarter_mean']:.3f}")
        print(f"  Last quarter: {prog['last_quarter_mean']:.3f}")
        print(f"  Improvement: {prog['improvement']:+.3f}")
    
    # Terminal observation distribution
    if "terminal_observations" in analysis:
        term_obs = analysis["terminal_observations"]
        print(f"\nTerminal Observation Distribution:")
        print(f"  Min n: {term_obs['min_n']}")
        print(f"  Max n: {term_obs['max_n']}")
        print(f"  Mean n: {term_obs['mean_n']:.2f}")
        print(f"  Terminals at n=0: {term_obs['terminals_at_n0']}")
        print(f"  Terminals explored: {term_obs['terminals_explored']}/{len(COMPONENT_TERMINALS)}")
    
    # Best system
    if "best_system" in analysis:
        best = analysis["best_system"]
        print(f"\nBest System:")
        print(f"  Name: {best['name']}")
        print(f"  Found at iteration: {best['discovered_at_iteration']}")
        print(f"  Phase: {best['phase']}")
        print(f"  Validation reward: {best['reward']:.3f}")
        print(f"  Components: {best['components']}")
    
    # Component coverage
    if "component_coverage" in results:
        print(f"\nComponent Coverage:")
        for phase, coverage in results["component_coverage"].items():
            print(f"  {phase}: {coverage['covered_components']}/{coverage['total_components']} "
                  f"({coverage['coverage_percentage']:.1f}%)")
            if coverage.get('uncovered_list'):
                print(f"    Uncovered: {coverage['uncovered_list']}")
    
    # Forcing efficiency
    if "forcing_efficiency" in analysis:
        force_eff = analysis["forcing_efficiency"]
        print(f"\nForcing Efficiency:")
        print(f"  Mean attempts: {force_eff['mean_attempts']:.1f}")
        print(f"  Median attempts: {force_eff['median_attempts']:.0f}")
        print(f"  Range: {force_eff['min_attempts']}-{force_eff['max_attempts']}")
    
    # Duplicate rate
    if "duplicate_rate" in results:
        print(f"\nDuplicate Rate: {results['duplicate_rate']:.1%}")
    
    # Grammar exploration
    sampler_stats = results.get("sampler_stats", {})
    if sampler_stats and "explored_production_rules" in sampler_stats:
        print(f"\nGrammar Exploration:")
        print(f"  Production rules explored: {sampler_stats['explored_production_rules']}/{sampler_stats['total_production_rules']} "
              f"({sampler_stats['exploration_percentage']:.1f}%)")
        print(f"  Unique sequences generated: {sampler_stats.get('unique_sequences_generated', 'N/A')}")
    
    # Test evaluation results
    if "test_evaluation" in results:
        test = results["test_evaluation"]
        
        if args.top_systems == 1 and isinstance(test, dict) and "test_accuracy" in test:
            # Single system format
            print(f"\nTest Set Performance:")
            print(f"  Dataset: {test.get('dataset_type', args.dataset)}")
            print(f"  Problems: {test.get('num_problems', 'N/A')}")
            print(f"  Runs per problem: {test.get('num_runs', 'N/A')}")
            print(f"  Test accuracy: {test['test_accuracy']:.1%} ± {test.get('test_std_error', 0.0):.1%}")
            print(f"  Execution rate: {test.get('execution_rate', 0.0):.1%}")
            if test.get('from_cache'):
                print(f"  Result source: CACHED")
            
        elif isinstance(test, dict) and "individual_results" in test:
            # Multi-system format
            print(f"\nTest Set Performance ({test['num_systems_tested']} systems):")
            for sys_name, result in test.get('individual_results', {}).items():
                if 'error' not in result:
                    cache_marker = " [cached]" if result.get('from_cache') else ""
                    print(f"  {sys_name}{cache_marker}:")
                    print(f"    Test accuracy: {result['test_accuracy']:.1%} ± {result.get('test_std_error', 0.0):.1%}")
                    print(f"    Execution rate: {result.get('execution_rate', 0.0):.1%}")
                else:
                    print(f"  {sys_name}: FAILED - {result['error']}")
            
            if test.get('best_test_system'):
                print(f"\n  Best test system: {test['best_test_system']} ({test['best_test_accuracy']:.1%})")


def save_results(executor, args, status="completed"):
    """Save experiment results with appropriate naming."""
    if status == "interrupted":
        output_file = f"interrupted_{args.output}" if args.output else None
    elif status == "error":
        output_file = f"error_{args.output}" if args.output else None
    else:
        output_file = args.output
    
    try:
        saved_file = executor.save_results(output_file)
        print(f"Results saved to: {saved_file}")
        return saved_file
    except Exception as e:
        print(f"Failed to save results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run simplified forced component curriculum Grammar Search experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core experiment parameters
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="Total iteration budget"
    )
    
    parser.add_argument(
        "--forced-iterations",
        type=int,
        default=30,
        help="Number of iterations for forced exploration phase"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["math", "gpqa", "aime", "musique", "mmlupro"],
        default="math",
        help="Dataset for evaluation"
    )
    
    # Length constraints
    parser.add_argument(
        "--forced-max-length",
        type=int,
        default=6,
        help="Maximum sequence length during forced exploration"
    )
    
    parser.add_argument(
        "--free-max-length",
        type=int,
        default=8,
        help="Maximum sequence length during free exploration"
    )
    
    # Sampling configuration for forced phase
    parser.add_argument(
        "--forced-sampling",
        type=str,
        choices=["thompson", "random"],
        default="thompson",
        help="Sampling method for forced phase"
    )
    
    parser.add_argument(
        "--no-forced-weights",
        action="store_true",
        help="Disable production weights during forced phase"
    )
    
    # Sampling configuration for free phase
    parser.add_argument(
        "--free-sampling",
        type=str,
        choices=["thompson", "random"],
        default="thompson",
        help="Sampling method for free phase"
    )
    
    parser.add_argument(
        "--no-free-weights",
        action="store_true",
        help="Disable production weights during free phase"
    )
    
    # Beta scores display
    parser.add_argument(
        "--no-beta-scores",
        action="store_true",
        help="Don't print Beta scores after each iteration"
    )
    
    parser.add_argument(
        "--beta-top-n",
        type=int,
        default=10,
        help="Number of top production rules to show in Beta scores"
    )
    
    # Thompson sampling parameters
    parser.add_argument(
        "--credit",
        type=str,
        choices=["full", "decay", "bottleneck"],
        default="bottleneck",
        help="Credit assignment strategy"
    )
    
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=1.0,
        help="Initial alpha for Beta prior"
    )
    
    parser.add_argument(
        "--beta-init",
        type=float,
        default=1.0,
        help="Initial beta for Beta prior"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--problems", "-p",
        type=int,
        default=32,
        help="Number of problems for validation evaluation"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=4,
        help="Evaluation runs per validation problem"
    )
    
    parser.add_argument(
        "--test-runs",
        type=int,
        default=4,
        help="Evaluation runs per test problem"
    )
    
    parser.add_argument(
        "--top-systems",
        type=int,
        default=1,
        help="Number of top systems to evaluate on test set"
    )
    
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip test set evaluation"
    )
    
    # External cache configuration
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable external evaluation cache"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for external cache (default: ~/.fusion_adas_cache)"
    )
    
    parser.add_argument(
        "--method-version",
        type=str,
        default="v1",
        help="Method version for cache isolation (e.g., v1, v2_pruning)"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache entries for this method version before running"
    )
    
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Print cache statistics and exit"
    )
    
    # Output configuration
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file for console output (default: auto-generated)"
    )
    
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory for debug logs (default: simplified_forced_<dataset>_debug_logs)"
    )
    
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)

    # Handle cache statistics request
    if args.cache_stats:
        from grammar_search.external_cache import ExternalEvaluationCache
        cache = ExternalEvaluationCache(cache_dir=args.cache_dir, 
                                       method_version=args.method_version)
        cache.print_stats()
        sys.exit(0)
    
    # Handle cache clearing request
    if args.clear_cache:
        from grammar_search.external_cache import ExternalEvaluationCache
        cache = ExternalEvaluationCache(cache_dir=args.cache_dir,
                                       method_version=args.method_version)
        response = input(f"Clear cache entries for method version '{args.method_version}'? (y/N): ")
        if response.lower() == 'y':
            count = cache.clear_cache(method_version=args.method_version)
            print(f"Cleared {count} cache entries")
        else:
            print("Cancelled")
        sys.exit(0)
    
    # Validate iteration budget
    forced_iterations, free_iterations = validate_iteration_budget(args)
    
    # Set up logging
    log_filename = setup_logging(args)
    
    # Run the main experiment with output tee'd to file
    with TeeOutput(log_filename):
        # Print header
        num_components = len(COMPONENT_TERMINALS)
        print_experiment_header(args, num_components)
        
        # Set up debug directory
        setup_debug_directory(args)
        
        # Create executor
        executor = SimplifiedForcedCurriculumExecutor(
            dataset_type=args.dataset,
            total_iterations=args.iterations,
            forced_iterations=args.forced_iterations,
            forced_max_length=args.forced_max_length,
            free_max_length=args.free_max_length,
            # Sampling configuration
            forced_sampling=args.forced_sampling,
            forced_use_weights=not args.no_forced_weights,
            free_sampling=args.free_sampling,
            free_use_weights=not args.no_free_weights,
            # Thompson parameters
            credit_assignment=args.credit,
            alpha_init=args.alpha_init,
            beta_init=args.beta_init,
            # Evaluation configuration
            num_problems=args.problems,
            num_eval_runs=args.runs,
            run_test_evaluation=not args.no_test,
            test_eval_runs=args.test_runs,
            num_top_systems=args.top_systems,
            # Beta scores display
            print_beta_scores=not args.no_beta_scores,
            beta_scores_top_n=args.beta_top_n,
            # External cache configuration
            use_external_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            method_version=args.method_version
        )
        
        try:
            # Run the experiment
            results = executor.run()
            
            # Print comprehensive summary
            print_experiment_summary(results, args)
            
            # Print cache summary
            executor.print_cache_summary()
            
            # Save results
            saved_file = save_results(executor, args, status="completed")
            
            print(f"\n{'='*60}")
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            if saved_file:
                print(f"Results file: {saved_file}")
            print(f"Log file: {log_filename}")
            print(f"Completed at: {datetime.now().isoformat()}")
            
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("EXPERIMENT INTERRUPTED BY USER")
            print("="*60)
            
            # Try to save partial results
            saved_file = save_results(executor, args, status="interrupted")
            
            if saved_file:
                print(f"Partial results saved: {saved_file}")
            print(f"Log file: {log_filename}")
            print(f"Interrupted at: {datetime.now().isoformat()}")
            sys.exit(1)
        
        except Exception as e:
            print(f"\n" + "="*60)
            print("EXPERIMENT FAILED WITH ERROR")
            print("="*60)
            print(f"Error: {e}")
            
            import traceback
            traceback.print_exc()
            
            # Try to save any partial results
            saved_file = save_results(executor, args, status="error")
            
            if saved_file:
                print(f"Partial results saved: {saved_file}")
            print(f"Log file: {log_filename}")
            print(f"Failed at: {datetime.now().isoformat()}")
            sys.exit(1)


if __name__ == "__main__":
    main()
