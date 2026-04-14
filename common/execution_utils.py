"""
Execution utilities for cluster evolution system.
"""

import ast
import time
import threading
import copy
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import namedtuple, Counter, defaultdict
import random
import re
import math

# Import LLM interface and debug logger
from common.config import BACKBONE_MODEL
from common.llm_interface import get_json_response_from_gpt, get_backbone_client, get_openai_api_key
import openai
from common.debug_logger import debug_logger

# Core data structure
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])


class LoopAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze loops in code for safety violations."""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.violations = []
        
    def visit_While(self, node):
        """Called when a While node is encountered."""
        self.violations.append({
            "type": "while_loop",
            "line": node.lineno,
            "message": f"while loop forbidden on line {node.lineno}"
        })
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Called when a For node is encountered."""
        # Check if it's a range() call that we can analyze
        if (isinstance(node.iter, ast.Call) and 
            isinstance(node.iter.func, ast.Name) and 
            node.iter.func.id == 'range'):
            
            # Check range arguments
            args = node.iter.args
            if len(args) == 1:
                # range(n) format
                if isinstance(args[0], ast.Constant) and isinstance(args[0].value, int):
                    if args[0].value > self.max_iterations:
                        self.violations.append({
                            "type": "excessive_iterations",
                            "line": node.lineno,
                            "message": f"for loop with {args[0].value} iterations exceeds limit of {self.max_iterations} on line {node.lineno}"
                        })
            elif len(args) >= 2:
                # range(start, stop) or range(start, stop, step) format
                start = 0
                if isinstance(args[0], ast.Constant) and isinstance(args[0].value, int):
                    start = args[0].value
                
                if isinstance(args[1], ast.Constant) and isinstance(args[1].value, int):
                    stop = args[1].value
                    iterations = max(0, stop - start)
                    if iterations > self.max_iterations:
                        self.violations.append({
                            "type": "excessive_iterations", 
                            "line": node.lineno,
                            "message": f"for loop with {iterations} iterations exceeds limit of {self.max_iterations} on line {node.lineno}"
                        })
        
        self.generic_visit(node)


class SystemValidator:
    """Validates system code for safety using AST analysis."""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
    
    def validate_system_code(self, system: Dict) -> Tuple[bool, str]:
        """
        Validate system code for safety issues using AST analysis.
        
        Args:
            system: Dictionary containing system with 'code' field
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        code = system.get("code", "")
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Check for loop violations using AST visitor
            analyzer = LoopAnalyzer(self.max_iterations)
            analyzer.visit(tree)
            
            if analyzer.violations:
                # Create detailed error message
                violation_details = []
                for violation in analyzer.violations:
                    violation_details.append(violation["message"])
                
                error_message = f"SAFETY_ERROR: {'; '.join(violation_details)}"
                return False, error_message
                
        except SyntaxError as e:
            return False, f"SYNTAX_ERROR: Invalid Python syntax - {str(e)}"
        except Exception as e:
            return False, f"PARSE_ERROR: Could not parse code for safety validation - {str(e)}"
        
        return True, ""

def extract_boxed(text: str) -> str:
    """
    Extract content from \\boxed{} in text.
    Returns the boxed content if found, otherwise returns the full text.
    """
    if not text:
        return text
    
    # Look for \\boxed{...} pattern
    # Handle nested braces by counting
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last boxed content (most likely the final answer)
        return matches[-1].strip()
    
    # Fallback: return full text if no boxed content found
    return text

class LLMAgentBase:
    """Base class for an LLM agent that can interact with language models."""

    def __init__(
        self,
        agent_name: str,
        role: str = 'helpful assistant',
        model: str = BACKBONE_MODEL,
        temperature: float = 0.5
    ) -> None:
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = self._random_id()

    def _random_id(self, length: int = 4) -> str:
        """Generate a random alphanumeric ID."""
        import random
        import string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_prompt(self, input_infos, instruction) -> tuple[str, str]:
        """Generate system prompt and user prompt from input information and instruction."""
        
        # Simple system prompt - just the role
        system_prompt = f"You are a {self.role}. Always put your final answer in \\boxed{{}}."

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                field_name, author, content, iteration_idx = input_info
            else:
                continue
            
            if author == self.__repr__():
                author += ' (yourself)'
            
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif field_name == 'answer':
                # For multi-agent communication, show previous answers
                if iteration_idx != -1:
                    input_infos_text += f'### Response #{iteration_idx + 1} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### Response by {author}:\n{content}\n\n'
            else:
                # Handle other info types generically
                if iteration_idx != -1:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction: str, iteration_idx: int = -1) -> Info:
        """Query the LLM and return response as single Info object."""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)

        try:
            # Use direct text response
            from common.llm_interface import get_text_response_from_gpt
            
            response = get_text_response_from_gpt(
                prompt, self.model, system_prompt, self.temperature,
                agent_info=f"{self.agent_name} (iter {iteration_idx})"
            )
            
            # Return as Info object with full response
            return Info('answer', self.__repr__(), response, iteration_idx)

        except Exception as e:
            # Return error as Info object
            return Info('answer', self.__repr__(), f"ERROR: {str(e)}", iteration_idx)

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction: str, iteration_idx: int = -1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem:
    """Container class for dynamically executing agent architectures."""
    
    def forward(self, taskInfo):
        """Process input and return final answer."""
        pass


class ThreadSafeAgentExecutor:
    """Thread-safe agent execution utility."""
    
    @staticmethod
    def create_thread_safe_agent_class(system_code: str, class_name_suffix: str = "") -> type:
        """
        Create a thread-safe AgentSystem subclass with the given code.
        
        Args:
            system_code: Python code string containing the forward function
            class_name_suffix: Optional suffix for the class name for debugging
            
        Returns:
            A new AgentSystem subclass with the forward method attached
        """
        # Set up execution environment
        execution_globals = {
            # Core classes
            'LLMAgentBase': LLMAgentBase,
            'Info': Info,
            'Counter': Counter,
            
            # ADD THESE TO ALLOW DIRECT LLM ACCESS:
            'openai': openai,
            'get_openai_api_key': get_openai_api_key,
            'get_backbone_client': get_backbone_client,
            'BACKBONE_MODEL': BACKBONE_MODEL,
            
            # Helper for boxed extraction
            'extract_boxed': extract_boxed,

            # Safe data processing
            'json': json,
            'math': math,
            're': re,
            
            # Safe randomization
            'random': random,
            
            # Safe data structures  
            'defaultdict': defaultdict,
            
            # Type annotations (runtime no-op)
            'List': List,
            'Dict': Dict,
            'Any': Any,
            'Tuple': Tuple,
            'Union': Union,
            'Optional': Optional,
            
            # Built-ins (already filtered by Python)
            '__builtins__': __builtins__
        }
        
        # Execute the system code to get the forward function
        namespace = {}
        exec(system_code, execution_globals, namespace)
        
        # Find the forward function
        forward_func = None
        for name, obj in namespace.items():
            if callable(obj) and name.startswith('forward'):
                forward_func = obj
                break
        
        if forward_func is None:
            raise ValueError("No forward function found in the provided code")
        
        # Create a unique AgentSystem subclass for this thread/execution
        thread_id = threading.get_ident()
        class_name = f"ThreadSafeAgentSystem_{thread_id}_{class_name_suffix}"
        
        ThreadSafeAgentSystem = type(class_name, (AgentSystem,), {})
        
        # Attach forward method to the thread-specific subclass only
        setattr(ThreadSafeAgentSystem, "forward", forward_func)
        
        return ThreadSafeAgentSystem
    
    @staticmethod
    def execute_system_safely(system: Dict, task_input: Any, 
                            extract_answer: callable = None) -> Dict:
        """
        Safely execute a system with thread isolation and error handling.
        
        Args:
            system: Dictionary containing system definition with 'code' and 'name'
            task_input: Input for the system (usually Info object or string)
            extract_answer: Optional function to extract answer from result
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Create thread-safe agent class
            agent_class = ThreadSafeAgentExecutor.create_thread_safe_agent_class(
                system["code"], 
                system.get("name", "unknown")
            )
            
            # Prepare task info if needed
            if isinstance(task_input, str):
                task_info = Info('task', 'User', task_input, -1)
            else:
                task_info = task_input
            
            # Execute with error handling
            try:
                agent_instance = agent_class()
                result = agent_instance.forward(task_info)
            except Exception as e:
                print("EXECUTION_ERROR", str(e))
                return {
                    "answer": "EXECUTION_ERROR",
                    "success": False,
                    "system_name": system.get("name", "unknown"),
                    "error": str(e)
                }
            
            # Handle error result
            if result == "ERROR":
                print("ERROR", result)
                return {
                    "answer": "ERROR",
                    "success": False,
                    "system_name": system.get("name", "unknown")
                }
            
            # Extract answer content
            if extract_answer:
                answer_content = extract_answer(result)
            elif hasattr(result, 'content'):
                # Extract boxed content from Info object's content
                full_content = result.content
                answer_content = extract_boxed(full_content)
            else:
                # Direct string result - extract boxed content
                answer_content = extract_boxed(str(result))
            
            return {
                "answer": answer_content,
                "success": True,
                "system_name": system.get("name", "unknown")
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {system.get('name', 'unknown')}",
                "success": False,
                "system_name": system.get("name", "unknown"),
                "error": str(e)
            }

def test_system_execution(system: Dict, test_question: str) -> Dict:
    """
    Test system execution with safety validation.
    
    Args:
        system: System definition dictionary
        test_question: Question to test the system on
        
    Returns:
        Dictionary with execution results and execution trace
    """
    if not test_question:
        return {"success": False, "error": "No test question provided"}
    
    # Safety check before execution using AST validator
    validator = SystemValidator(max_iterations=10)
    is_safe, safety_error = validator.validate_system_code(system)
    if not is_safe:
        return {
            "success": False,
            "error": safety_error,
            "execution_trace": "Code validation failed - execution skipped for safety"
        }
        
    try:
        # Create a simple test problem
        test_info = Info('task', 'User', test_question, -1)
        
        def extract_full_result(result):
            """Extract both thinking and answer if available."""
            if hasattr(result, 'content'):
                # If it's a single Info object, just return content
                return result.content
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                # If it's a tuple/list (thinking, answer), extract both
                thinking_info, answer_info = result[0], result[1]
                thinking = thinking_info.content if hasattr(thinking_info, 'content') else str(thinking_info)
                answer = answer_info.content if hasattr(answer_info, 'content') else str(answer_info)
                return {"thinking": thinking, "answer": answer}
            else:
                # Fallback to string representation
                return str(result)
        
        result = ThreadSafeAgentExecutor.execute_system_safely(
            system, 
            test_info,
            extract_answer=extract_full_result
        )
        
        if result["success"]:
            answer_data = result["answer"]
            if isinstance(answer_data, dict) and "thinking" in answer_data:
                execution_trace = f"THINKING: {answer_data['thinking']}\nFINAL ANSWER: {answer_data['answer']}"
            else:
                execution_trace = f"Successfully executed and produced: {answer_data}"
            
            return {
                "success": True,
                "answer": answer_data,
                "execution_trace": execution_trace
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "execution_trace": "System failed to execute"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_trace": f"Exception during testing: {str(e)}"
        }
