#!/usr/bin/env python3
"""
Grammar rules and terminals for grammar-based system generation.
"""

from typing import Dict, List, Tuple
import random

MODULAR_GRAMMAR_RULES = {
    # System must end with a single output, but the very first block
    # must not be SelfCriticIteration because it expects prior context.
    "System": [
        "StartSingleOutput"
    ],

    # StartSingleOutput: first position rules
    # Allowed first items:
    #   - SingleInputSingleOutput that does NOT include SelfCriticIteration
    #   - SingleInputMultiOutput (branch into multi immediately)
    # After the first item, hand off to the regular SingleOutput / MultiInput flows.
    "StartSingleOutput": [
        "StartSingleInputSingleOutput",                         # end after one safe SISO
        "StartSingleInputSingleOutput SingleOutput",            # continue chaining in SingleOutput
        "SingleInputMultiOutput MultiInput"                     # start with a fan-out and go to MultiInput
    ],

    # SingleOutput: general single-flow after position 1
    # Here SelfCriticIteration is allowed because we already have context.
    "SingleOutput": [
        "SingleInputSingleOutput",                              # end with single output
        "SingleInputSingleOutput SingleOutput",                 # keep chaining single→single
        "SingleInputMultiOutput MultiInput"                     # branch to multi flow
    ],

    # MultiInput: cannot stop, must eventually collapse back to SingleOutput
    "MultiInput": [
        "MultiInputSingleOutput",                               # collapse immediately, end with single
        "MultiInputSingleOutput SingleOutput",                  # collapse, then continue chaining in SingleOutput
        "MultiInputMultiOutput MultiInput"                      # stay in multi flow
    ],

    # Interface categories expanded
    # First-position SISO that explicitly EXCLUDES SelfCriticIteration
    "StartSingleInputSingleOutput": [
        "StepByStepReasonerSingleOutput",
        "RoleBasedReasonerSingleOutput"
    ],

    # General SISO (allowed after the first position)
    # This INCLUDES SelfCriticIteration variants
    "SingleInputSingleOutput": [
        "StepByStepReasonerSingleOutput",
        "RoleBasedReasonerSingleOutput",
        "SelfCriticIteration"
    ],

    "StepByStepReasonerSingleOutput": [
        "StepByStepReasoner(count=1)"
    ],

    "RoleBasedReasonerSingleOutput": [
        "RoleBasedReasoner(count=1)"
    ],

    # SelfCriticIteration requires prior context, so it is only reachable
    # through SingleInputSingleOutput, not through StartSingleInputSingleOutput.
    "SelfCriticIteration": [
        "SelfCriticIteration(rounds=5)"
    ],

    "SingleInputMultiOutput": [
        "StepByStepReasonerMultiOutput",
        "RoleBasedReasonerMultiOutput"
    ],

    "StepByStepReasonerMultiOutput": [
        "StepByStepReasoner(count=5)"
    ],

    "RoleBasedReasonerMultiOutput": [
        "RoleBasedReasoner(count=5)"
    ],

    "MultiInputSingleOutput": [
        "MajorityVoter",
        "ConsensusBuilder"
    ],

    "MultiInputMultiOutput": [
        "DebateIteration",
        "MultiSelfCriticIteration"
    ],

    "DebateIteration": [
        "DebateIteration(rounds=2)"
    ],
    
    "MultiSelfCriticIteration": [
        "MultiSelfCriticIteration(rounds=5)"
    ]
}

# Explicitly mark all component terminals in the grammar
COMPONENT_TERMINALS = {
    # StepByStepReasoner variants
    "StepByStepReasoner(count=1)",
    "StepByStepReasoner(count=5)",
    
    # RoleBasedReasoner variants
    "RoleBasedReasoner(count=1)",
    "RoleBasedReasoner(count=5)",
    
    # SelfCriticIteration variants
    "SelfCriticIteration(rounds=5)",
    
    # DebateIteration variants
    "DebateIteration(rounds=2)",
    
    # MultiSelfCriticIteration variants
    "MultiSelfCriticIteration(rounds=5)",
    
    # Voting/Consensus components
    "MajorityVoter",
    "ConsensusBuilder"
}


class GrammarSampler:
    """Basic grammar sampler for deriving sequences (minimal functionality needed)."""
    
    def __init__(self, grammar_rules: Dict[str, List[str]]):
        self.grammar_rules = grammar_rules
    
    def is_terminal(self, symbol: str) -> bool:
        """Check if a symbol is terminal (not in grammar rules)."""
        return symbol not in self.grammar_rules
    
    def sample_production(self, symbol: str) -> str:
        """Sample a production rule for the given symbol."""
        if symbol not in self.grammar_rules:
            return symbol
        
        productions = self.grammar_rules[symbol]
        return random.choice(productions)
    
    def derive_sequence(self, start_symbol: str = "System") -> Tuple[List[str], List[str]]:
        """
        Derive a complete sequence from the grammar.
        
        Returns:
            Tuple of (derivation_steps, terminal_components)
        """
        derivation_steps = [start_symbol]
        current_symbols = [start_symbol]
        
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while current_symbols and iteration < max_iterations:
            iteration += 1
            new_symbols = []
            
            for symbol in current_symbols:
                if self.is_terminal(symbol):
                    new_symbols.append(symbol)
                else:
                    # Sample a production for this non-terminal
                    production = self.sample_production(symbol)
                    derivation_steps.append(f"{symbol} → {production}")
                    
                    # Split the production into individual symbols
                    production_symbols = production.split()
                    new_symbols.extend(production_symbols)
            
            current_symbols = new_symbols
            
            # Check if all symbols are terminal
            if all(self.is_terminal(s) for s in current_symbols):
                break
        
        # Extract terminal components (those with parentheses are components)
        terminal_components = [s for s in current_symbols if '(' in s or s in ['MajorityVoter', 'ConsensusBuilder']]
        
        return derivation_steps, terminal_components
