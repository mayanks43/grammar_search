"""
Simple Azure OpenAI utilities for retry logic.
Handles Azure's exact retry-after timing and common API errors.
"""

import re
import time
import random
import openai
import logging

from common.cancellation_utils import interruptible_sleep


def extract_retry_after_seconds(error_message: str) -> int:
    """
    Extract retry-after time from Azure OpenAI error message.
    
    Examples:
    "Please retry after 3 seconds" -> 3
    "Please retry after 44 seconds" -> 44
    """
    if not error_message:
        return 0
        
    # Look for "retry after X seconds" pattern
    match = re.search(r'retry after (\d+) seconds?', error_message, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def azure_retry_backoff(max_tries: int = 10):
    """
    Retry decorator for Azure OpenAI API calls with intelligent backoff.
    
    Handles:
    - Rate limit errors with Azure's exact retry-after timing
    - Connection/timeout/server errors with short retry
    - Max token errors with short retry
    - Invalid judge responses with short retry
    
    Usage:
        @azure_retry_backoff(max_tries=10)
        def my_llm_function():
            # Your LLM call here
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_tries):
                try:
                    return func(*args, **kwargs)
                    
                except openai.RateLimitError as e:
                    last_exception = e
                    
                    # Extract exact wait time from Azure error message
                    retry_after_seconds = extract_retry_after_seconds(str(e))
                    
                    if retry_after_seconds > 0:
                        # Use Azure's exact timing
                        logging.info(f"Rate limit retry {attempt + 1}/{max_tries}: waiting {retry_after_seconds}s as requested by Azure")
                        wait_time = retry_after_seconds
                    else:
                        # Fallback if no retry-after found
                        logging.info(f"Rate limit retry {attempt + 1}/{max_tries}: no retry-after found, waiting 2s")
                        wait_time = 2.0
                    
                    if not interruptible_sleep(wait_time):
                        raise TimeoutError("Task cancelled during rate limit retry")
                
                # Group connection and server errors together
                except (openai.APIConnectionError, 
                        openai.APITimeoutError, 
                        openai.NotFoundError, 
                        openai.InternalServerError) as e:
                    last_exception = e
                    error_type = type(e).__name__
                    
                    # Short wait for transient errors
                    wait_time = 2.0
                    logging.info(f"{error_type} retry {attempt + 1}/{max_tries}: waiting {wait_time}s")
                    
                    if not interruptible_sleep(wait_time):
                        raise TimeoutError(f"Task cancelled during {error_type} retry")
                
                except openai.BadRequestError as e:
                    error_message = str(e)
                    
                    # Only retry for specific BadRequest scenarios
                    if any(keyword in error_message.lower() for keyword in ["max_tokens", "output limit", "maximum context", "content filtering"]):
                        last_exception = e
                        wait_time = 1.5 * (attempt + 1)
                        logging.info(f"Token limit retry {attempt + 1}/{max_tries}: waiting {wait_time}s")

                        if not interruptible_sleep(wait_time):
                            raise TimeoutError("Task cancelled during token limit retry")
                    else:
                        # Other BadRequest errors shouldn't be retried
                        print(f"Non-retryable BadRequestError: {e}")
                        raise
                
                except ValueError as e:
                    error_message = str(e)
                    
                    # Only retry for judge response validation errors
                    if any(keyword in error_message.lower() for keyword in ["invalid judge response", "expecting value", "response truncated"]):
                        last_exception = e
                        wait_time = 1.0

                        if "invalid judge response" in error_message.lower():
                            logging.info(f"Invalid judge response retry {attempt + 1}/{max_tries}: waiting {wait_time}s")
                        elif "expecting value" in error_message.lower():
                            logging.info(f"Expecting value error retry {attempt + 1}/{max_tries}: waiting {wait_time}s")
                        else:
                            logging.info(f"Response truncated error retry {attempt + 1}/{max_tries}: waiting {wait_time}s")
                        
                        if not interruptible_sleep(wait_time):
                            raise TimeoutError("Task cancelled during judge response retry")
                    else:
                        # Other ValueErrors shouldn't be retried
                        print(f"Non-retryable ValueError: {e}")
                        raise
                
                except TimeoutError:
                    # Don't retry task cancellations
                    raise
                
                except Exception as e:
                    # Don't retry unexpected exceptions
                    print(f"Non-retryable exception {type(e).__name__}: {e}")
                    raise
            
            # All retries exhausted
            print(f"All {max_tries} retries exhausted for LLM call. Last error: {last_exception}")
            print(args, kwargs)
            raise last_exception
        
        return wrapper
    return decorator
