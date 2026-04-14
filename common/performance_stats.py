"""
Unified system performance calculation module with pass@k metrics.
Consolidates performance calculation logic and confidence intervals.
Updated to include both std_error and std_dev in return values.
Uses expanded terminology in fitness strings.
"""

from typing import List, Dict, Tuple, NamedTuple, Optional
import numpy as np
import scipy.stats as stats
from collections import defaultdict


class SystemPerformanceStats(NamedTuple):
    """Structured container for system performance statistics."""
    mean_accuracy: float
    std_error: float  # Standard error of the mean
    std_dev: float    # Standard deviation of the data
    min_accuracy: float
    max_accuracy: float
    num_runs: int
    per_run_accuracies: List[float]
    confidence_interval: str
    execution_times: List[float]
    avg_execution_time: float
    # Pass@k metrics
    pass_at_k: Optional[float] = None
    pass_at_k_std_error: Optional[float] = None
    per_problem_passes: Optional[List[float]] = None


def calculate_pass_at_k(results: List[Dict], k: int) -> Tuple[float, float, List[float]]:
    """
    Calculate pass@k metric.
    
    Args:
        results: List of evaluation results with problem_idx, run_id, is_correct
        k: Number of runs
    
    Returns:
        Tuple of (pass_rate, std_error, per_problem_passes)
    """
    # Group results by problem
    problem_results = defaultdict(list)
    for result in results:
        problem_idx = result.get("problem_idx", result.get("problem_index", -1))
        if problem_idx >= 0:
            problem_results[problem_idx].append(result.get("is_correct", False))
    
    # Calculate pass/fail for each problem
    per_problem_passes = []
    for problem_idx in sorted(problem_results.keys()):
        # Did at least one run solve this problem?
        passed = any(problem_results[problem_idx])
        per_problem_passes.append(1.0 if passed else 0.0)
    
    if not per_problem_passes:
        return 0.0, 0.0, []
    
    # Calculate pass rate
    pass_rate = np.mean(per_problem_passes)
    
    # Calculate standard error for proportion
    n_problems = len(per_problem_passes)
    if n_problems > 1:
        std_error = np.sqrt(pass_rate * (1 - pass_rate) / n_problems)
    else:
        std_error = 0.0
    
    return pass_rate, std_error, per_problem_passes


def calculate_confidence_interval(
    run_accuracies: List[float],
    confidence_level: float = 0.95
) -> str:
    """
    Calculate confidence interval with expanded terminology for clarity.
    
    Args:
        run_accuracies: List of per-run accuracies
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Formatted string with expanded terminology
    """
    if not run_accuracies:
        return "No data"
    
    # Calculate avg@k metrics
    data = np.array(run_accuracies)
    n = len(data)
    confidence_pct = int(confidence_level * 100)
    
    if n == 1:
        mean_perf = data[0]
        ci_str = f"{confidence_pct}% Confidence Interval: ({mean_perf*100:.1f}%, {mean_perf*100:.1f}%), Mean: {mean_perf*100:.1f}%, Standard Error: 0.0%, Standard Deviation: 0.0%, n=1"
    else:
        mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        std_error = sample_std / np.sqrt(n)
        
        # Calculate t-critical value and margin of error using standard error
        alpha = 1 - confidence_level
        degrees_freedom = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df=degrees_freedom)
        margin_of_error = t_critical * std_error
        
        # Calculate confidence interval bounds
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
        ci_str = (f"{confidence_pct}% Confidence Interval: ({lower_bound*100:.1f}%, {upper_bound*100:.1f}%), "
                  f"Mean: {mean*100:.1f}%, Standard Error: {std_error*100:.1f}%, Standard Deviation: {sample_std*100:.1f}%, n={n}")
    
    return ci_str


def calculate_system_performance(
    evaluation_results: List[Dict], 
    confidence_level: float = 0.95,
    include_pass_at_k: bool = False
) -> SystemPerformanceStats:
    """
    Calculate system performance using proper per-run averaging methodology with optional pass@k.
    Returns both std_error and std_dev for different use cases.
    
    Args:
        evaluation_results: List of evaluation results. Each result must contain:
            - run_id: Unique identifier for the evaluation run
            - is_correct: Boolean indicating if the answer was correct
            - execution_time: Time taken for execution (optional)
            - problem_idx or problem_index: Problem identifier (optional, for pass@k)
        confidence_level: Confidence level for t-distribution interval (default: 0.95)
        include_pass_at_k: Whether to calculate pass@k metrics
        
    Returns:
        SystemPerformanceStats: Complete performance statistics with both std_error and std_dev
    """
    assert evaluation_results, "evaluation_results cannot be empty"
    
    # Validate required fields exist in all results
    required_fields = {"run_id", "is_correct"}
    for i, result in enumerate(evaluation_results):
        missing_fields = required_fields - set(result.keys())
        assert not missing_fields, f"Result {i} missing required fields: {missing_fields}"
    
    # Group results by run_id for per-run calculation
    run_groups = defaultdict(list)
    execution_times = []
    
    for result in evaluation_results:
        run_id = result["run_id"]
        is_correct = result["is_correct"]
        
        assert isinstance(is_correct, bool), f"is_correct must be boolean, got {type(is_correct)}"
        
        run_groups[run_id].append(is_correct)
        
        # Collect execution times if available
        exec_time = result.get("execution_time")
        execution_times.append(exec_time or 0.0)
    
    assert run_groups, "No run groups found - run_id field may be invalid"
    
    # Calculate accuracy for each run
    per_run_accuracies = []
    
    for run_id in sorted(run_groups.keys()):
        run_results = run_groups[run_id]
        
        assert run_results, f"Run {run_id} has no evaluation results"
        
        # Calculate accuracy for this specific run
        correct_count = sum(run_results)
        total_count = len(run_results)
        run_accuracy = correct_count / total_count
        
        per_run_accuracies.append(run_accuracy)
    
    assert per_run_accuracies, "No per-run accuracies calculated"
    
    # Calculate core statistics using per-run averaging
    mean_accuracy = np.mean(per_run_accuracies)
    
    # Calculate both std_dev and std_error
    if len(per_run_accuracies) > 1:
        std_dev = np.std(per_run_accuracies, ddof=1)  # Sample standard deviation
        std_error = std_dev / np.sqrt(len(per_run_accuracies))  # Standard error of the mean
    else:
        std_dev = 0.0
        std_error = 0.0
    
    min_accuracy = np.min(per_run_accuracies)
    max_accuracy = np.max(per_run_accuracies)
    
    # Calculate pass@k metrics if requested
    pass_at_k = None
    pass_at_k_std_error = None
    per_problem_passes = None
    
    if include_pass_at_k:
        k = len(per_run_accuracies)
        pass_at_k, pass_at_k_std_error, per_problem_passes = calculate_pass_at_k(evaluation_results, k)
    
    # Calculate confidence interval
    confidence_interval = calculate_confidence_interval(
        per_run_accuracies,
        confidence_level
    )
    
    # Calculate execution time statistics
    avg_execution_time = np.mean(execution_times) if execution_times else 0.0
    
    return SystemPerformanceStats(
        mean_accuracy=float(mean_accuracy),
        std_error=float(std_error),
        std_dev=float(std_dev),
        min_accuracy=float(min_accuracy),
        max_accuracy=float(max_accuracy),
        num_runs=len(per_run_accuracies),
        per_run_accuracies=[float(acc) for acc in per_run_accuracies],
        confidence_interval=confidence_interval,
        execution_times=[float(t) for t in execution_times],
        avg_execution_time=float(avg_execution_time),
        pass_at_k=float(pass_at_k) if pass_at_k is not None else None,
        pass_at_k_std_error=float(pass_at_k_std_error) if pass_at_k_std_error is not None else None,
        per_problem_passes=[float(p) for p in per_problem_passes] if per_problem_passes else None
    )


def select_top_systems_statistical(
    systems_with_stats: List[Tuple[Dict, float, float, float, int]], 
    max_systems: int = 5,
    alpha: float = 0.05
) -> List[Dict]:
    """
    Select top systems using Welch's t-test for statistical hypothesis testing.
    Common function used by both ADAS and Grammar Search sampling approaches.
    Note: Uses standard error for hypothesis testing (not std_dev).
    
    Args:
        systems_with_stats: List of tuples (system_dict, mean, std_error, std_dev, n)
            where system_dict is the original system dictionary,
            mean is the performance mean, std_error is standard error,
            std_dev is standard deviation, and n is the sample size
        max_systems: Maximum number of systems to select
        alpha: Significance level for hypothesis testing (default 0.05)
    
    Returns:
        List of selected system dictionaries
    """
    from scipy import stats
    
    if not systems_with_stats:
        print("No valid systems found for test evaluation")
        return []
    
    # Sort by mean performance descending
    systems_with_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Get the best system
    best_system, best_mean, best_std_error, best_std_dev, best_n = systems_with_stats[0]
    
    # Convert std_error back to std for Welch's test
    best_std = best_std_error * np.sqrt(best_n)
    
    print(f"\nBest system: {best_system.get('name', 'Unknown')}")
    print(f"  Mean: {best_mean:.3f}, Standard Error: {best_std_error:.3f}, Standard Deviation: {best_std_dev:.3f}, n={best_n}")
    print(f"\nUsing Welch's t-test with α={alpha}")
    
    # Select systems using Welch's t-test
    selected_systems = [best_system]  # Always include the best system
    
    for system, sys_mean, sys_std_error, sys_std_dev, sys_n in systems_with_stats[1:]:
        # Convert std_error back to std for Welch's test
        sys_std = sys_std_error * np.sqrt(sys_n)
        
        # Perform Welch's t-test
        # H0: system mean = best system mean
        # H1: system mean ≠ best system mean (two-tailed)
        
        # Handle edge cases where std is 0
        if best_std == 0 and sys_std == 0:
            # If both have zero variance, compare means directly
            if abs(best_mean - sys_mean) < 1e-10:
                p_value = 1.0  # Essentially identical
            else:
                p_value = 0.0  # Clearly different
        else:
            # Use Welch's t-test
            t_stat, p_value = stats.ttest_ind_from_stats(
                best_mean, best_std, best_n,
                sys_mean, sys_std, sys_n,
                equal_var=False  # Welch's t-test
            )
        
        print(f"  Testing {system.get('name', 'Unknown')}: Mean={sys_mean:.3f}, p-value={p_value:.4f}", end="")
        
        if p_value > alpha:
            # Fail to reject H0 - system is not significantly different from best
            selected_systems.append(system)
            print(" → Not significantly different (selected)")
        else:
            print(" → Significantly different (not selected)")
    
    # Limit to max_systems
    if len(selected_systems) > max_systems:
        print(f"\nLimiting to top {max_systems} systems by mean performance")
        selected_systems = selected_systems[:max_systems]
    
    # Print final selection
    print(f"\nSelected {len(selected_systems)} systems for test evaluation:")
    for i, system in enumerate(selected_systems, 1):
        # Find the stats for this system
        for sys, mean, std_error, std_dev, n in systems_with_stats:
            if sys == system:
                print(f"  {i}. {system.get('name', 'Unknown')}: Mean={mean:.3f}, Standard Error={std_error:.3f}, Standard Deviation={std_dev:.3f}, n={n}")
                break
    
    return selected_systems


def parse_fitness_string(fitness_str: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    """
    Parse fitness string to extract statistics including both std_error and std_dev.
    Uses expanded format only.
    
    Args:
        fitness_str: String like "95% Confidence Interval: (71.2%, 78.8%), Mean: 75.0%, Standard Error: 1.75%, Standard Deviation: 3.5%, n=5"
    
    Returns:
        Tuple of (mean, std_error, std_dev, n) or (None, None, None, None) if parsing fails
    """
    import re
    
    try:
        # Extract mean
        mean_match = re.search(r'Mean:\s*(\d+\.?\d*)%?', fitness_str)
        if mean_match:
            mean = float(mean_match.group(1)) / 100  # Convert percentage to decimal
        else:
            return None, None, None, None
        
        # Extract standard error (expanded format only)
        se_match = re.search(r'Standard Error:\s*(\d+\.?\d*)%?', fitness_str)
        if se_match:
            std_error = float(se_match.group(1)) / 100
        else:
            std_error = None
        
        # Extract standard deviation (expanded format only)
        sd_match = re.search(r'Standard Deviation:\s*(\d+\.?\d*)%?', fitness_str)
        if sd_match:
            std_dev = float(sd_match.group(1)) / 100
        else:
            std_dev = None
        
        # Extract sample size
        n_match = re.search(r'n=(\d+)', fitness_str)
        if n_match:
            n = int(n_match.group(1))
        else:
            n = 1  # Default to 1 if not found
        
        # If we have std_dev but not std_error, calculate it
        if std_dev is not None and std_error is None and n > 0:
            std_error = std_dev / np.sqrt(n)
        
        # If we have std_error but not std_dev, calculate it
        if std_error is not None and std_dev is None and n > 0:
            std_dev = std_error * np.sqrt(n)
        
        return mean, std_error, std_dev, n
    except:
        return None, None, None, None


def extract_system_stats_grammar(system: Dict) -> Tuple[float, float, float, int]:
    """
    Extract statistics from Grammar Search sampling system format.
    Returns both std_error and std_dev.
    
    Args:
        system: System dict with 'reward' and 'full_results'
    
    Returns:
        Tuple of (mean, std_error, std_dev, n)
    """
    import numpy as np
    
    full_results = system.get('full_results', {})
    
    mean = system.get('reward', 0.0)
    std_error = system.get('std_error', 0.0)
    std_dev = system.get('std_dev', std_error * np.sqrt(full_results.get('num_runs', 1)))
    n = full_results.get('num_runs', 1)

    return mean, std_error, std_dev, n


def select_systems_for_test_adas(archive: List[Dict], max_systems: int = 5, alpha: float = 0.05) -> List[Dict]:
    """
    ADAS wrapper: Select systems from archive with fitness strings.
    
    Args:
        archive: List of systems with 'fitness' field containing formatted strings
        max_systems: Maximum number of systems to select
        alpha: Significance level
    
    Returns:
        List of selected systems
    """
    # Parse all systems and create tuples
    systems_with_stats = []
    
    for system in archive:
        if 'fitness' in system and system['fitness'] != "No data":
            mean, std_error, std_dev, n = parse_fitness_string(system['fitness'])
            if mean is not None:
                # Ensure we have both std_error and std_dev
                if std_error is None and std_dev is not None:
                    std_error = std_dev / np.sqrt(n) if n > 0 else 0.0
                if std_dev is None and std_error is not None:
                    std_dev = std_error * np.sqrt(n)
                
                systems_with_stats.append((system, mean, std_error, std_dev, n))
    
    # Use common selection function
    return select_top_systems_statistical(systems_with_stats, max_systems, alpha)


def select_systems_for_test_grammar(all_systems: List[Dict], num_systems: int, alpha: float = 0.05) -> List[Dict]:
    """
    Grammar Search wrapper: Select systems from all_systems with reward/full_results.
    
    Args:
        all_systems: List of systems with 'reward' and 'full_results' fields
        num_systems: Maximum number of systems to select
        alpha: Significance level
    
    Returns:
        List of selected systems
    """
    # Extract stats from all systems
    systems_with_stats = []
    
    for system in all_systems:
        if 'reward' in system and system['reward'] > 0:
            mean, std_error, std_dev, n = extract_system_stats_grammar(system)
            systems_with_stats.append((system, mean, std_error, std_dev, n))
    
    # Use common selection function
    return select_top_systems_statistical(systems_with_stats, num_systems, alpha)
