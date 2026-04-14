#!/usr/bin/env python3
"""
Weighted grammar configuration for biasing toward complex multi-agent architectures.
"""

from grammar_search.grammar_rules import MODULAR_GRAMMAR_RULES

# Production weights - each key must exist in MODULAR_GRAMMAR_RULES 
# and have exactly the same number of weights as productions
PRODUCTION_WEIGHTS = {
    # StartSingleOutput has 3 productions - bias toward multi-agent
    "StartSingleOutput": [
        0.1,  # "StartSingleInputSingleOutput" - simple termination
        0.4,  # "StartSingleInputSingleOutput SingleOutput" - single with continuation  
        0.5   # "SingleInputMultiOutput MultiInput" - multi-agent
    ],
    
    # SingleOutput has 3 productions - balanced between single paths and branching
    "SingleOutput": [
        0.2,  # "SingleInputSingleOutput"
        0.3,  # "SingleInputSingleOutput SingleOutput"
        0.5   # "SingleInputMultiOutput MultiInput"
    ]
}
