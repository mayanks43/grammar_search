#!/usr/bin/env python3
"""
Simplified Forced Component Curriculum Sampler for Grammar Search.
Two-phase approach: Forced exploration with strict n-ordering, then free Thompson/Random sampling.
Removed pruning and multiple forced phases.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

from grammar_search.grammar_rules import MODULAR_GRAMMAR_RULES, COMPONENT_TERMINALS
from grammar_search.weighted_grammar_config import PRODUCTION_WEIGHTS
from grammar_search.template_generator import IsolatedTemplateGenerator


@dataclass
class ThompsonStats:
    """Statistics for Thompson Sampling using Beta-Binomial conjugate prior."""
    alpha: float = 1.0  # Success prior
    beta: float = 1.0   # Failure prior
    n: int = 0          # Number of observations
    
    def update(self, reward: float):
        """Update statistics with new reward (assumes reward in [0,1])."""
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.n += 1
    
    def sample(self) -> float:
        """Sample from Beta posterior distribution."""
        return np.random.beta(self.alpha, self.beta)
    
    def get_mean(self) -> float:
        """Get mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)


class SimplifiedForcedCurriculumSampler:
    """Simplified Thompson sampler with forced exploration based on observation counts."""
    
    def __init__(self, 
                 grammar_rules: Dict[str, List[str]] = None,
                 production_weights: Dict[str, List[float]] = None,
                 alpha_init: float = 1.0,
                 beta_init: float = 1.0,
                 dataset_type: str = None):
        """
        Initialize simplified forced curriculum sampler.
        
        Args:
            grammar_rules: Grammar production rules
            production_weights: Weights for each production rule
            alpha_init: Initial alpha for Beta prior
            beta_init: Initial beta for Beta prior
            dataset_type: Type of dataset for code generation
        """
        self.grammar_rules = grammar_rules or MODULAR_GRAMMAR_RULES
        self.production_weights = production_weights or {}
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        
        # Pre-compute non-terminals for efficient checking
        self.non_terminals = set(self.grammar_rules.keys())
        
        # Track statistics for each production rule
        self.production_stats = {}  # (non_terminal, production) -> ThompsonStats
        
        # Current n threshold for forced exploration
        self.current_n_threshold = 0  # Current exploration level
        
        # Frozen systems cache
        self.frozen_systems = {}  # sequence_tuple -> system_dict
        
        # Code generator
        self.code_generator = IsolatedTemplateGenerator(dataset_type)
        
        # Track component coverage per phase
        self.phase_component_coverage = defaultdict(lambda: defaultdict(int))
        
        # Sampling mode configuration
        self.sampling_mode = "thompson"  # "thompson" or "random"
        self.use_weights = True
        
        # Derivation cache for analysis
        self.derivation_cache = {}
    
    def get_terminal_observation_count(self, terminal: str) -> int:
        """
        Get observation count (n) for a terminal from Beta statistics.
        This finds the production rule that generates this terminal and returns its n.
        
        Args:
            terminal: Terminal component name
            
        Returns:
            Observation count from Beta statistics (0 if never observed)
        """
        # Find the production rule that generates this terminal
        for (non_terminal, production), stats in self.production_stats.items():
            # Check if this production contains the terminal
            if terminal in production.split():
                return stats.n
        
        # Terminal has never been observed
        return 0
    
    def get_terminals_at_threshold(self, n_threshold: int) -> List[str]:
        """Get all terminals with exactly n observations based on Beta statistics."""
        terminals_at_n = []
        
        for terminal in COMPONENT_TERMINALS:
            if self.get_terminal_observation_count(terminal) == n_threshold:
                terminals_at_n.append(terminal)
        
        return terminals_at_n
    
    def get_terminal_performance(self) -> Dict[str, float]:
        """
        Get performance scores for each terminal component based on Beta distribution means.
        
        Returns:
            Dictionary mapping terminal component to its performance score
        """
        terminal_scores = {}
        
        # For each terminal, find the production rule that generates it
        for (non_terminal, production), stats in self.production_stats.items():
            # Check each symbol in the production
            for symbol in production.split():
                if symbol in COMPONENT_TERMINALS:
                    # This production generates a terminal component
                    terminal_scores[symbol] = stats.get_mean()
        
        return terminal_scores
    
    def is_component(self, symbol: str) -> bool:
        """Check if symbol is a component terminal."""
        return symbol in COMPONENT_TERMINALS
    
    def is_non_terminal(self, symbol: str) -> bool:
        """Check if symbol is a non-terminal."""
        return symbol in self.non_terminals
    
    def sample_production(self, non_terminal: str) -> str:
        """
        Sample a production for a non-terminal based on current sampling mode.
        
        Args:
            non_terminal: The non-terminal symbol
            
        Returns:
            The chosen production string
        """
        productions = self.grammar_rules[non_terminal]
        
        if self.sampling_mode == "random":
            # Random sampling with weights if available
            if self.use_weights and non_terminal in self.production_weights:
                weights = self.production_weights[non_terminal]
                # Normalize weights in case they don't sum to 1
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w/total_weight for w in weights]
                    return np.random.choice(productions, p=normalized_weights)
                else:
                    return random.choice(productions)
            else:
                return random.choice(productions)
        
        else:  # thompson sampling
            # Get weights if available
            weights = self.production_weights.get(non_terminal, [1.0] * len(productions)) if self.use_weights else [1.0] * len(productions)
            
            # Get Thompson samples for each production
            samples = []
            for production in productions:
                key = (non_terminal, production)
                
                if key not in self.production_stats:
                    self.production_stats[key] = ThompsonStats(
                        alpha=self.alpha_init,
                        beta=self.beta_init
                    )
                
                thompson_sample = self.production_stats[key].sample()
                samples.append(thompson_sample)
            
            # Combine Thompson samples with weights
            combined_scores = [sample * weight for sample, weight in zip(samples, weights)]
            
            # Sample proportionally to combined scores
            total_score = sum(combined_scores)
            if total_score == 0:
                return random.choice(productions)
            
            probabilities = [score / total_score for score in combined_scores]
            chosen_idx = np.random.choice(len(productions), p=probabilities)
            return productions[chosen_idx]
    
    def sample_sequence_with_derivation(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Sample component sequence using current sampling mode.
        
        Returns:
            (component_sequence, derivation_path)
            where derivation_path is [(non_terminal, chosen_production), ...]
        """
        derivation_path = []
        current_symbols = ["System"]
        components = []
        
        max_iterations = 50  # Safety limit
        iteration = 0
        
        while current_symbols and iteration < max_iterations:
            iteration += 1
            symbol = current_symbols.pop(0)
            
            if self.is_non_terminal(symbol):
                # Non-terminal - sample production
                production = self.sample_production(symbol)
                
                # Record the choice
                derivation_path.append((symbol, production))
                
                # Add production symbols to process
                production_symbols = production.split()
                # Add to front of list to maintain order
                current_symbols = production_symbols + current_symbols
                
            elif self.is_component(symbol):
                # It's a component we want to track
                components.append(symbol)
            # else: it's some other terminal we ignore
        
        # Cache the derivation for analysis
        sequence_key = tuple(components)
        self.derivation_cache[sequence_key] = derivation_path
        
        return components, derivation_path
    
    def sample_with_max_length(self, max_length: int, max_attempts: int = 100) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Sample sequence with length <= max_length.
        
        Args:
            max_length: Maximum allowed sequence length
            max_attempts: Maximum sampling attempts before fallback
            
        Returns:
            (component_sequence, derivation_path)
        """
        for attempt in range(max_attempts):
            components, derivation = self.sample_sequence_with_derivation()
            if len(components) <= max_length:
                return components, derivation
        
        # Fallback: find shortest possible sequence
        print(f"⚠️  WARNING: Struggled to find sequence ≤ {max_length} after {max_attempts} attempts")
        best = None
        best_len = float('inf')
        
        for _ in range(20):
            components, derivation = self.sample_sequence_with_derivation()
            if len(components) < best_len:
                best_len = len(components)
                best = (components, derivation)
        
        if best:
            print(f"   Using fallback sequence of length {best_len}")
            return best
        
        # Emergency fallback - should never happen
        print("⚠️  CRITICAL: Could not generate any valid sequence")
        emergency_derivation = [
            ("System", "StartSingleOutput"),
            ("StartSingleOutput", "StartSingleInputSingleOutput"),
            ("StartSingleInputSingleOutput", "StepByStepReasonerSingleOutput"),
            ("StepByStepReasonerSingleOutput", "StepByStepReasoner(count=1)")
        ]
        return ["StepByStepReasoner(count=1)"], emergency_derivation
    
    def force_component_with_max_length(self, 
                                       target_component: str, 
                                       max_length: int,
                                       max_attempts: int = 1000) -> Tuple[List[str], List[Tuple[str, str]], int]:
        """
        Force a specific component to appear in a sequence with max length constraint.
        
        Args:
            target_component: Component that must appear in the sequence
            max_length: Maximum allowed sequence length
            max_attempts: Maximum attempts before giving up
            
        Returns:
            (component_sequence, derivation_path, num_attempts)
        """
        # Direct repeated sampling with two conditions
        for attempt in range(1, max_attempts + 1):
            components, derivation = self.sample_sequence_with_derivation()
            
            # Check both conditions: length AND component presence
            if len(components) <= max_length and target_component in components:
                return components, derivation, attempt
        
        # Failed to force component after max_attempts
        return [], [], max_attempts
    
    def set_sampling_mode(self, mode: str, use_weights: bool):
        """
        Configure sampling mode.
        
        Args:
            mode: 'thompson' or 'random'
            use_weights: Whether to use production weights
        """
        self.sampling_mode = mode
        self.use_weights = use_weights
    
    def get_or_generate_system(self, component_sequence: List[str]) -> Dict[str, any]:
        """
        Get frozen system for sequence, generating if needed.
        
        Args:
            component_sequence: List of components
            
        Returns:
            System dictionary with code and metadata
        """
        sequence_key = tuple(component_sequence)
        
        if sequence_key in self.frozen_systems:
            return self.frozen_systems[sequence_key]
        
        # Generate new system
        auto_name = f"System_{len(self.frozen_systems):03d}"
        generated_code = self.code_generator.generate_code(component_sequence, auto_name)
        
        # Extract and clean up the name
        llm_name = generated_code.get("name", "UnnamedSystem")
        if llm_name.startswith(auto_name):
            # Remove auto-name prefix if LLM included it
            llm_name = llm_name[len(auto_name):].strip()
            if llm_name.startswith(":"):
                llm_name = llm_name[1:].strip()
        
        # Combine auto-generated name with LLM name
        combined_name = f"{auto_name}: {llm_name}" if llm_name else auto_name
        
        system = {
            "name": combined_name,
            "thought": generated_code.get("analysis", "Generated system"),
            "code": generated_code.get("code", "def forward(self, taskInfo): return taskInfo"),
            "component_sequence": component_sequence,
            "frozen": True,
            "generation_timestamp": np.datetime64('now').astype(float)
        }
        
        self.frozen_systems[sequence_key] = system
        return system
    
    def update_stats(self, derivation_path: List[Tuple[str, str]], reward: float,
                     credit_assignment: str = "bottleneck"):
        """
        Update statistics for all production rules used in this derivation.
        
        Args:
            derivation_path: [(non_terminal, production), ...] from sample_sequence_with_derivation
            reward: Observed reward [0, 1]
            credit_assignment: Strategy for credit assignment
        """
        if credit_assignment == "full":
            # Full credit to all productions
            for non_terminal, production in derivation_path:
                key = (non_terminal, production)
                
                if key not in self.production_stats:
                    self.production_stats[key] = ThompsonStats(
                        alpha=self.alpha_init,
                        beta=self.beta_init
                    )
                
                self.production_stats[key].update(reward)
        
        elif credit_assignment == "decay":
            # Decay credit along the path
            decay = 0.9
            current_reward = reward
            
            for non_terminal, production in reversed(derivation_path):
                key = (non_terminal, production)
                
                if key not in self.production_stats:
                    self.production_stats[key] = ThompsonStats(
                        alpha=self.alpha_init,
                        beta=self.beta_init
                    )
                
                self.production_stats[key].update(current_reward)
                current_reward *= decay
        
        elif credit_assignment == "bottleneck":
            # Only update non-deterministic choices
            for non_terminal, production in derivation_path:
                # Only update if there were multiple choices
                if len(self.grammar_rules[non_terminal]) > 1:
                    key = (non_terminal, production)
                    
                    if key not in self.production_stats:
                        self.production_stats[key] = ThompsonStats(
                            alpha=self.alpha_init,
                            beta=self.beta_init
                        )
                    
                    self.production_stats[key].update(reward)
    
    def update_component_coverage(self, phase: str, components: List[str]):
        """Update coverage tracking for a phase."""
        for component in components:
            if component in COMPONENT_TERMINALS:
                self.phase_component_coverage[phase][component] += 1
    
    def get_stats_summary(self) -> Dict:
        """Get comprehensive summary of current statistics."""
        # Production rule performance
        production_performance = []
        
        for (non_terminal, production), stats in self.production_stats.items():
            production_performance.append({
                "non_terminal": non_terminal,
                "production": production,
                "mean_reward": stats.get_mean(),
                "alpha": stats.alpha,
                "beta": stats.beta,
                "observations": stats.n,
                "confidence": 1.0 / (1.0 + np.sqrt(stats.n)) if stats.n > 0 else 0.0
            })
        
        production_performance.sort(key=lambda x: x["mean_reward"], reverse=True)
        
        # Grammar coverage analysis
        total_productions = sum(len(prods) for prods in self.grammar_rules.values())
        explored_productions = len(self.production_stats)
        
        # Non-terminal coverage
        non_terminal_coverage = defaultdict(lambda: {"explored": 0, "total": 0})
        
        for non_terminal, productions in self.grammar_rules.items():
            non_terminal_coverage[non_terminal]["total"] = len(productions)
            
            for production in productions:
                if (non_terminal, production) in self.production_stats:
                    non_terminal_coverage[non_terminal]["explored"] += 1
        
        # Terminal observation summary based on Beta statistics
        terminal_obs = {}
        for terminal in COMPONENT_TERMINALS:
            terminal_obs[terminal] = self.get_terminal_observation_count(terminal)
        
        obs_values = list(terminal_obs.values())
        terminal_obs_summary = {
            "min_observations": min(obs_values) if obs_values else 0,
            "max_observations": max(obs_values) if obs_values else 0,
            "mean_observations": np.mean(obs_values) if obs_values else 0,
            "observation_distribution": terminal_obs
        }
        
        return {
            "total_production_rules": total_productions,
            "explored_production_rules": explored_productions,
            "exploration_percentage": (explored_productions / total_productions * 100) if total_productions > 0 else 0,
            "total_systems_generated": len(self.frozen_systems),
            "top_productions": production_performance[:20],
            "non_terminal_coverage": dict(non_terminal_coverage),
            "unique_sequences_generated": len(self.derivation_cache),
            "sampling_mode": self.sampling_mode,
            "weights_enabled": self.use_weights,
            "terminal_observations": terminal_obs_summary,
            "current_n_threshold": self.current_n_threshold
        }
    
    def get_component_coverage_stats(self) -> Dict:
        """Get statistics about component coverage across phases."""
        coverage = {}
        
        all_components = set(COMPONENT_TERMINALS)
        
        for phase, component_counts in self.phase_component_coverage.items():
            covered = set(component_counts.keys())
            uncovered = all_components - covered
            
            coverage[phase] = {
                "total_components": len(all_components),
                "covered_components": len(covered),
                "uncovered_components": len(uncovered),
                "coverage_percentage": (len(covered) / len(all_components)) * 100,
                "uncovered_list": list(uncovered),
                "usage_counts": dict(component_counts)
            }
        
        return coverage
    
    def print_phase_summary(self, phase_name: str):
        """Print summary statistics for a completed phase."""
        print(f"\n{'='*50}")
        print(f"Phase {phase_name} Summary")
        print(f"{'='*50}")
        
        # Coverage stats
        if phase_name in self.phase_component_coverage:
            coverage = self.get_component_coverage_stats().get(phase_name, {})
            if coverage:
                print(f"Component Coverage: {coverage['covered_components']}/{coverage['total_components']} "
                      f"({coverage['coverage_percentage']:.1f}%)")
                
                if coverage['uncovered_list']:
                    print(f"Uncovered components: {coverage['uncovered_list']}")
        
        # Terminal observation distribution for forced phase (from Beta statistics)
        if phase_name == "forced_exploration":
            print(f"\nTerminal Observation Distribution (from Beta statistics):")
            obs_counts = defaultdict(int)
            for terminal in COMPONENT_TERMINALS:
                n = self.get_terminal_observation_count(terminal)
                obs_counts[n] += 1
            
            for n in sorted(obs_counts.keys()):
                print(f"  n={n}: {obs_counts[n]} terminals")
            
            print(f"  Current n threshold: {self.current_n_threshold}")
        
        print(f"Sampling mode: {self.sampling_mode} {'with' if self.use_weights else 'without'} weights")
