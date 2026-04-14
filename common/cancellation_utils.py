"""
Cancellation utilities for interruptible operations.
Minimal addition to support timeout-aware sleeps.
"""

import time
import threading
from typing import Optional


# Thread-local storage for cancellation events
_thread_local = threading.local()


def set_task_cancellation_event(event: threading.Event):
    """Set cancellation event for current thread."""
    _thread_local.cancellation_event = event


def get_task_cancellation_event() -> Optional[threading.Event]:
    """Get cancellation event for current thread, if any."""
    return getattr(_thread_local, 'cancellation_event', None)


def clear_task_cancellation_event():
    """Clear cancellation event for current thread."""
    if hasattr(_thread_local, 'cancellation_event'):
        delattr(_thread_local, 'cancellation_event')


def interruptible_sleep(duration: float, check_interval: float = 0.5) -> bool:
    """
    Sleep for duration but wake up periodically to check for cancellation.
    
    Args:
        duration: Total sleep duration in seconds
        check_interval: How often to check for cancellation (default: 0.5s)
    
    Returns:
        True if sleep completed normally
        False if cancelled early
    """
    if duration <= 0:
        return True
    
    cancellation_event = get_task_cancellation_event()
    if cancellation_event is None:
        # No cancellation event set, use regular sleep
        time.sleep(duration)
        return True
    
    end_time = time.time() + duration
    
    while time.time() < end_time:
        if cancellation_event.is_set():
            return False  # Cancelled
        
        remaining = end_time - time.time()
        sleep_duration = min(check_interval, remaining)
        
        if sleep_duration > 0:
            time.sleep(sleep_duration)
    
    return True  # Completed normally
