"""
Debug logging utilities for cluster evolution system.
Extracted from adas/debug_utils.py with context-aware logging.
"""

import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path

# Configuration
DEBUG_LOGGING = True  # Set to True to enable debug logging
DEFAULT_DEBUG_LOG_DIR = "default_debug_logs"  # Default directory


class ClusterDebugLogger:
    """Handles debug logging for cluster evolution with organized folder structure."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.call_counter = 0
        
        # Thread-local storage for context
        self._thread_local = threading.local()
        
        # Configurable log directory
        self._debug_log_dir = DEFAULT_DEBUG_LOG_DIR
        self._log_dir_set = False  # Track if directory was explicitly set
    
    def set_log_directory(self, path: str):
        """
        Set the debug log directory. Must be called before any logging operations.
        
        Args:
            path: Directory path for debug logs
            
        Raises:
            RuntimeError: If logging has already started
        """
        if self.call_counter > 0 or self._log_dir_set:
            raise RuntimeError(
                "Cannot change log directory after logging has started. "
                "Call set_log_directory() before any logging operations."
            )
        
        self._debug_log_dir = path
        self._log_dir_set = True
        
        # Create directory if it doesn't exist
        os.makedirs(self._debug_log_dir, exist_ok=True)
        print(f"Debug log directory set to: {self._debug_log_dir}")
    
    def get_log_directory(self) -> str:
        """Get the current debug log directory."""
        return self._debug_log_dir
    
    @property
    def DEBUG_LOG_DIR(self) -> str:
        """Property for backward compatibility with existing code."""
        return self._debug_log_dir
    
    def _get_thread_context(self):
        """Get thread-local context, initializing if needed."""
        if not hasattr(self._thread_local, 'context_type'):
            self._thread_local.context_type = None
            self._thread_local.architecture_name = None
            self._thread_local.problem_id = None
            self._thread_local.accumulated_calls = []
        return self._thread_local
    
    def start_architecture_generation(self, architecture_name: str):
        """Set context for architecture generation."""
        if not DEBUG_LOGGING:
            return
        
        context = self._get_thread_context()
        context.context_type = "generation"
        context.architecture_name = architecture_name
        context.problem_id = None
        context.accumulated_calls = []
    
    def end_architecture_generation(self):
        """End generation context and write accumulated logs to file."""
        if not DEBUG_LOGGING:
            return
        
        context = self._get_thread_context()
        if context.context_type != "generation" or not context.architecture_name:
            return
        
        try:
            # Create generation folder structure
            generation_dir = os.path.join(self._debug_log_dir, "generation", context.architecture_name)
            os.makedirs(generation_dir, exist_ok=True)
            
            # Write all accumulated calls to single file
            filepath = os.path.join(generation_dir, "generation_complete.json")
            
            log_data = {
                "metadata": {
                    "session_id": self.session_id,
                    "architecture_name": context.architecture_name,
                    "timestamp": datetime.now().isoformat(),
                    "context_type": "generation",
                    "total_calls": len(context.accumulated_calls)
                },
                "llm_calls": context.accumulated_calls
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Clear context
            context.context_type = None
            context.architecture_name = None
            context.accumulated_calls = []
            
        except Exception as e:
            print(f"Failed to write generation log: {e}")
    
    def start_architecture_evaluation(self, architecture_name: str):
        """Set context for architecture evaluation (called once per architecture)."""
        if not DEBUG_LOGGING:
            return
        
        # This just sets the architecture context, problems will set problem_id
        context = self._get_thread_context()
        context.architecture_name = architecture_name
    
    def start_problem_evaluation(self, architecture_name: str, problem_id: int):
        """Set context for specific problem evaluation (called per thread/problem)."""
        if not DEBUG_LOGGING:
            return
        
        context = self._get_thread_context()
        context.context_type = "evaluation"
        context.architecture_name = architecture_name
        context.problem_id = problem_id
        context.accumulated_calls = []
    
    def end_problem_evaluation(self):
        """End problem context and write accumulated logs to file."""
        if not DEBUG_LOGGING:
            return
        
        context = self._get_thread_context()
        if context.context_type != "evaluation" or not context.architecture_name or context.problem_id is None:
            return
        
        try:
            # Create evaluation folder structure
            eval_dir = os.path.join(self._debug_log_dir, "evaluation", context.architecture_name)
            os.makedirs(eval_dir, exist_ok=True)
            
            # Write problem-specific log
            filepath = os.path.join(eval_dir, f"problem_{context.problem_id:04d}.json")
            
            log_data = {
                "metadata": {
                    "session_id": self.session_id,
                    "architecture_name": context.architecture_name,
                    "problem_id": context.problem_id,
                    "thread_id": threading.get_ident(),
                    "timestamp": datetime.now().isoformat(),
                    "context_type": "evaluation",
                    "total_calls": len(context.accumulated_calls)
                },
                "llm_calls": context.accumulated_calls
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Clear problem context (keep architecture context for thread reuse)
            context.context_type = None
            context.problem_id = None
            context.accumulated_calls = []
            
        except Exception as e:
            print(f"Failed to write problem evaluation log: {e}")
    
    def start_problem_judging(self, architecture_name: str, problem_id: int):
        """Set context for problem judging."""
        if not DEBUG_LOGGING:
            return
        
        context = self._get_thread_context()
        context.context_type = "judging"
        context.architecture_name = architecture_name
        context.problem_id = problem_id
        context.accumulated_calls = []
    
    def end_problem_judging(self):
        """End judging context and write accumulated logs to file."""
        if not DEBUG_LOGGING:
            return
        
        context = self._get_thread_context()
        if context.context_type != "judging" or not context.architecture_name or context.problem_id is None:
            return
        
        try:
            # Create evaluation folder structure (same as evaluation)
            eval_dir = os.path.join(self._debug_log_dir, "evaluation", context.architecture_name)
            os.makedirs(eval_dir, exist_ok=True)
            
            # Write judging-specific log
            filepath = os.path.join(eval_dir, f"problem_{context.problem_id:04d}_judging.json")
            
            log_data = {
                "metadata": {
                    "session_id": self.session_id,
                    "architecture_name": context.architecture_name,
                    "problem_id": context.problem_id,
                    "thread_id": threading.get_ident(),
                    "timestamp": datetime.now().isoformat(),
                    "context_type": "judging",
                    "total_calls": len(context.accumulated_calls)
                },
                "llm_calls": context.accumulated_calls
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Clear judging context
            context.context_type = None
            context.architecture_name = None
            context.problem_id = None
            context.accumulated_calls = []
            
        except Exception as e:
            print(f"Failed to write problem judging log: {e}")
    
    def log_llm_call(self, call_type: str, model: str, system_prompt: str, user_prompt: str, 
                     response: str, problem_info: str = None, agent_info: str = None):
        """
        Log an LLM call with context-aware routing.
        """
        if not DEBUG_LOGGING:
            return
        
        try:
            # Get thread context
            context = self._get_thread_context()
            
            # Create call log entry
            call_data = {
                "call_number": len(context.accumulated_calls) + 1,
                "timestamp": datetime.now().isoformat(),
                "call_type": call_type,
                "model": model,
                "agent_info": agent_info,
                "problem_info": problem_info,
                "llm_interaction": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response
                }
            }
            
            # Route based on context
            if context.context_type in ["generation", "evaluation", "judging"]:
                # Accumulate in thread-local storage
                context.accumulated_calls.append(call_data)
            else:
                # Fallback to original individual file behavior
                self._write_individual_log_file(call_type, model, system_prompt, user_prompt, 
                                              response, problem_info, agent_info)
                
        except Exception as e:
            print(f"Debug logging failed: {e}")
    
    def _write_individual_log_file(self, call_type: str, model: str, system_prompt: str, 
                                  user_prompt: str, response: str, problem_info: str = None, 
                                  agent_info: str = None):
        """Fallback to original individual file logging behavior."""
        try:
            # Create debug directory if it doesn't exist
            os.makedirs(self._debug_log_dir, exist_ok=True)
            
            # Increment call counter
            self.call_counter += 1
            
            # Create filename with session, counter, and call type
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"{self.session_id}_{self.call_counter:04d}_{timestamp}_{call_type}.json"
            filepath = os.path.join(self._debug_log_dir, filename)
            
            # Prepare log data
            log_data = {
                "metadata": {
                    "session_id": self.session_id,
                    "call_number": self.call_counter,
                    "timestamp": datetime.now().isoformat(),
                    "call_type": call_type,
                    "model": model,
                    "problem_info": problem_info,
                    "agent_info": agent_info
                },
                "llm_interaction": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response
                }
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Individual file logging failed: {e}")
    
    def log_problem_start(self, problem_number: int, problem_text: str, ground_truth: str = None):
        """Log the start of a new problem."""
        if not DEBUG_LOGGING:
            return
        
        try:
            context = self._get_thread_context()
            if context.context_type == "evaluation":
                # Add problem start info to accumulated calls
                start_data = {
                    "call_number": len(context.accumulated_calls) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "call_type": "problem_start",
                    "problem_data": {
                        "problem_number": problem_number,
                        "problem_text": problem_text,
                        "ground_truth": ground_truth
                    }
                }
                context.accumulated_calls.append(start_data)
                return
            
            # Fallback to original behavior
            self._write_individual_log_file("problem_start", "N/A", "", problem_text, ground_truth or "", 
                                          f"Problem {problem_number}", "Problem Logger")
                
        except Exception as e:
            print(f"Problem start logging failed: {e}")
    
    def log_problem_result(self, problem_number: int, final_answer: str, is_correct: bool, 
                          execution_time: float = None):
        """Log the final result of a problem."""
        if not DEBUG_LOGGING:
            return
        
        try:
            context = self._get_thread_context()
            if context.context_type == "evaluation":
                # Add result info to accumulated calls
                result_data = {
                    "call_number": len(context.accumulated_calls) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "call_type": "problem_result",
                    "result_data": {
                        "problem_number": problem_number,
                        "final_answer": final_answer,
                        "is_correct": is_correct,
                        "execution_time_seconds": execution_time
                    }
                }
                context.accumulated_calls.append(result_data)
                return
            
            # Fallback to original behavior
            result_text = f"Answer: {final_answer}, Correct: {is_correct}, Time: {execution_time}s"
            self._write_individual_log_file("problem_result", "N/A", "", result_text, "", 
                                          f"Problem {problem_number}", "Result Logger")
                
        except Exception as e:
            print(f"Problem result logging failed: {e}")


# Global debug logger instance
debug_logger = ClusterDebugLogger()

# For backward compatibility - this now returns the current directory dynamically
DEBUG_LOG_DIR = debug_logger.DEBUG_LOG_DIR
