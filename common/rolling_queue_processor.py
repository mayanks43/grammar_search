"""
Generic rolling queue processor for memory-efficient task execution.
Abstracts the common rolling queue pattern used across evaluators.
UPDATED: Added timeout-aware task cancellation support.
"""

import json
import time
import threading
from typing import Dict, List, Iterator, Tuple, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from queue import Queue, Empty
from pathlib import Path
import datetime

from common.cancellation_utils import set_task_cancellation_event, clear_task_cancellation_event

# Try to import tqdm, but don't fail if it's not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class SimpleProgressPrinter:
    """Simple progress printer for job environments."""
    
    def __init__(self, total_tasks: int, log_interval: int = 100):
        self.total_tasks = total_tasks
        self.log_interval = log_interval
        self.completed = 0
        self.fallback_count = 0  # Add fallback tracking
        self.timeout_count = 0   # Add timeout tracking
        self.cancelled_count = 0  # NEW: Add cancellation tracking
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        print(f"Progress tracking started: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"Total tasks: {total_tasks}, will print every {log_interval} completions")
    
    def update(self, increment: int = 1, fallback_triggered: bool = False, 
               timeout_triggered: bool = False, cancelled: bool = False):
        """Update progress and print if needed."""
        with self._lock:
            self.completed += increment
            if fallback_triggered:
                self.fallback_count += 1
            if timeout_triggered:
                self.timeout_count += 1
            if cancelled:
                self.cancelled_count += 1
            
            if self.completed % self.log_interval == 0 or self.completed == self.total_tasks:
                self._print_progress()
    
    def _print_progress(self):
        """Print progress update with fallback, timeout, and cancellation info."""
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total_tasks - self.completed) / rate if rate > 0 else 0
        
        progress_pct = (self.completed / self.total_tasks) * 100
        fallback_pct = (self.fallback_count / self.completed) * 100 if self.completed > 0 else 0
        timeout_pct = (self.timeout_count / self.completed) * 100 if self.completed > 0 else 0
        cancelled_pct = (self.cancelled_count / self.completed) * 100 if self.completed > 0 else 0
        
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
              f"{self.completed:>6}/{self.total_tasks} ({progress_pct:5.1f}%) | "
              f"Rate: {rate:5.1f}/s | "
              f"ETA: {eta/60:5.1f}m | "
              f"Fallbacks: {self.fallback_count} ({fallback_pct:.1f}%) | "
              f"Timeouts: {self.timeout_count} ({timeout_pct:.1f}%) | "
              f"Cancelled: {self.cancelled_count} ({cancelled_pct:.1f}%)")
    
    def close(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        fallback_pct = (self.fallback_count / self.completed) * 100 if self.completed > 0 else 0
        timeout_pct = (self.timeout_count / self.completed) * 100 if self.completed > 0 else 0
        cancelled_pct = (self.cancelled_count / self.completed) * 100 if self.completed > 0 else 0
        
        print(f"Progress completed: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Final rate: {self.completed/elapsed:.1f} tasks/second")
        print(f"Final fallback rate: {self.fallback_count} ({fallback_pct:.1f}%)")
        print(f"Final timeout rate: {self.timeout_count} ({timeout_pct:.1f}%)")
        print(f"Final cancellation rate: {self.cancelled_count} ({cancelled_pct:.1f}%)")


class RollingQueueProcessor:
    """Generic rolling queue processor for memory-efficient task execution with timeout support."""
    
    def __init__(self, 
                 queue_size: int = 200,
                 save_interval: int = 500,
                 checkpoint_file: str = None,
                 log_interval: int = 100,
                 use_progress_printing: bool = True,
                 task_timeout: int = 600):
        self.queue_size = queue_size
        self.save_interval = save_interval
        self.checkpoint_file = checkpoint_file
        self.log_interval = log_interval
        self.use_progress_printing = use_progress_printing
        self.task_timeout = task_timeout
        
        # Thread-safe state
        self._stop_event = threading.Event()
        
        # NEW: Task cancellation tracking
        self._active_futures = {}  # future -> cancellation_event
        self._futures_lock = threading.Lock()
        
    def _task_producer(self, task_queue: Queue, task_generator: Iterator, progress_tracker):
        """Producer thread that feeds tasks to the queue."""
        tasks_produced = 0
        try:
            for task_data in task_generator:
                if self._stop_event.is_set():
                    break
                
                task_queue.put(task_data)
                tasks_produced += 1
                
        except Exception as e:
            print(f"Task producer error after {tasks_produced} tasks: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                task_queue.put(None)  # Signal end of tasks
            except Exception as e:
                print(f"Error signaling end of tasks: {e}")
    
    def _save_checkpoint(self, completed_results: List[Dict], 
                        prepare_checkpoint_fn: Callable[[List[Dict]], Dict]):
        """Save progress checkpoint using provided preparation function."""
        if not self.checkpoint_file:
            return
            
        try:
            checkpoint_data = prepare_checkpoint_fn(completed_results)
            
            temp_file = str(self.checkpoint_file) + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            Path(temp_file).rename(self.checkpoint_file)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, validate_checkpoint_fn: Callable[[Dict], Tuple[bool, int, List[Dict]]]) -> Tuple[int, List[Dict]]:
        """Load progress checkpoint using provided validation function."""
        if not self.checkpoint_file or not Path(self.checkpoint_file).exists():
            return 0, []
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            is_valid, start_from, results = validate_checkpoint_fn(checkpoint_data)
            
            if not is_valid:
                print("Checkpoint validation failed, starting fresh")
                return 0, []
            
            print(f"Resuming from checkpoint: {start_from} tasks completed")
            return start_from, results
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return 0, []
    
    def _execute_task_with_cancellation(self, task_executor_fn: Callable[[Any], Dict], 
                                       task_data: Any) -> Dict:
        """Execute task with cancellation support."""
        # Create cancellation event for this task
        cancellation_event = threading.Event()
        
        # Set up cancellation context for the current thread
        set_task_cancellation_event(cancellation_event)
        
        try:
            result = task_executor_fn(task_data)
            return result
        finally:
            # Clean up cancellation context
            clear_task_cancellation_event()
    
    def process_with_rolling_queue(self,
                                 task_generator_fn: Callable[..., Iterator],
                                 task_executor_fn: Callable[[Any], Dict],
                                 prepare_checkpoint_fn: Callable[[List[Dict]], Dict],
                                 validate_checkpoint_fn: Callable[[Dict], Tuple[bool, int, List[Dict]]],
                                 create_error_result_fn: Callable[[Any, Exception], Dict],
                                 total_tasks: int,
                                 max_workers: int = 4,
                                 resume: bool = True,
                                 task_generator_args: Tuple = (),
                                 progress_desc: str = "Processing") -> List[Dict]:
        """
        Process tasks using rolling queue with memory efficiency and timeout support.
        
        Args:
            task_generator_fn: Function that returns task iterator
            task_executor_fn: Function that executes single task
            prepare_checkpoint_fn: Function that prepares checkpoint data
            validate_checkpoint_fn: Function that validates checkpoint and returns (valid, start_from, results)
            create_error_result_fn: Function that creates error result from task and exception
            total_tasks: Total number of tasks
            max_workers: Number of parallel workers
            resume: Whether to resume from checkpoint
            task_generator_args: Arguments for task generator function
            progress_desc: Description for progress tracking
        
        Returns:
            List of results
        """
        # Load checkpoint if resuming
        start_from = 0
        all_results = []
        if resume:
            start_from, checkpoint_results = self._load_checkpoint(validate_checkpoint_fn)
            all_results.extend(checkpoint_results)
            print(f"Continuing with {len(all_results)} previous results")
        
        remaining_tasks = total_tasks - start_from
        
        print(f"Processing {total_tasks} tasks with rolling queue (queue_size={self.queue_size})")
        print(f"Starting from task {start_from}")
        print(f"Task timeout: {self.task_timeout} seconds")
        print(f"Timeout support: Enhanced with cancellation-aware sleeps")
        
        if remaining_tasks == 0:
            print("All tasks already completed!")
            return all_results
        
        # Create task queue and generator
        task_queue = Queue(maxsize=self.queue_size)
        task_generator = task_generator_fn(start_from, *task_generator_args)
        
        # Initialize progress tracking
        progress_tracker = None
        pbar = None
        
        if self.use_progress_printing:
            progress_tracker = SimpleProgressPrinter(remaining_tasks, self.log_interval)
        elif TQDM_AVAILABLE:
            pbar = tqdm(total=remaining_tasks, desc=progress_desc, 
                       initial=0, unit="tasks", dynamic_ncols=True)
        
        try:
            # Start producer thread
            producer_thread = threading.Thread(
                target=self._task_producer,
                args=(task_queue, task_generator, progress_tracker),
                daemon=True
            )
            producer_thread.start()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                active_futures = {}
                
                # Fill initial queue
                for _ in range(min(max_workers * 2, self.queue_size)):
                    try:
                        task_data = task_queue.get(timeout=1.0)
                        if task_data is None:
                            break
                        
                        # Submit task with cancellation support
                        future = executor.submit(self._execute_task_with_cancellation, task_executor_fn, task_data)
                        
                        # Track future and its cancellation event
                        cancellation_event = threading.Event()
                        active_futures[future] = {
                            'task_data': task_data,
                            'cancellation_event': cancellation_event,
                            'start_time': time.time()
                        }
                        
                    except Empty:
                        break
                
                # Rolling execution with enhanced timeout handling
                while active_futures:
                    try:
                        # Add timeout to as_completed - this prevents hanging on stuck futures
                        completed = next(as_completed(active_futures, timeout=self.task_timeout + 10))
                        future_info = active_futures.pop(completed)
                        task_data = future_info['task_data']
                        start_time = future_info['start_time']
                        
                        try:
                            # Add timeout to result() - this kills individual stuck tasks
                            result = completed.result(timeout=self.task_timeout)
                            all_results.append(result)
                            
                            # Extract status info if available
                            fallback_triggered = result.get("fallback_triggered", False)
                            timeout_triggered = result.get("timeout", False)
                            cancelled = result.get("cancelled", False)
                            
                            # Update progress trackers
                            if progress_tracker:
                                progress_tracker.update(1, fallback_triggered, timeout_triggered, cancelled)
                            if pbar:
                                pbar.update(1)
                            
                            # Save checkpoint periodically
                            if len(all_results) % self.save_interval == 0:
                                self._save_checkpoint(all_results, prepare_checkpoint_fn)
                                
                        except TimeoutError:
                            execution_time = time.time() - start_time
                            print(f"Task timed out after {execution_time:.1f}s (limit: {self.task_timeout}s)")
                            
                            # Signal cancellation to the task
                            future_info['cancellation_event'].set()
                            
                            # Add timeout error result
                            timeout_error = TimeoutError(f"Task timed out after {self.task_timeout} seconds")
                            error_result = create_error_result_fn(task_data, timeout_error)
                            error_result["timeout"] = True
                            error_result["cancelled"] = True
                            all_results.append(error_result)
                            
                            # Update progress trackers
                            if progress_tracker:
                                progress_tracker.update(1, False, True, True)  # timeout + cancelled
                            if pbar:
                                pbar.update(1)
                                
                        except Exception as e:
                            print(f"Task failed: {e}")
                            # Add error result
                            error_result = create_error_result_fn(task_data, e)
                            all_results.append(error_result)
                            
                            # Update progress trackers
                            if progress_tracker:
                                progress_tracker.update(1, False, False, False)
                            if pbar:
                                pbar.update(1)
                        
                        # Try to get next task
                        try:
                            task_data = task_queue.get_nowait()
                            if task_data is None:
                                continue
                            
                            # Submit new task with cancellation support
                            future = executor.submit(self._execute_task_with_cancellation, task_executor_fn, task_data)
                            
                            # Track new future
                            cancellation_event = threading.Event()
                            active_futures[future] = {
                                'task_data': task_data,
                                'cancellation_event': cancellation_event,
                                'start_time': time.time()
                            }
                            
                        except Empty:
                            continue
                            
                    except TimeoutError:
                        # This catches timeout from as_completed - means all futures are stuck
                        print(f"All workers appear stuck, cancelling {len(active_futures)} futures")
                        for future, future_info in list(active_futures.items()):
                            future.cancel()
                            
                            # Signal cancellation
                            future_info['cancellation_event'].set()
                            
                            task_data = future_info['task_data']
                            timeout_error = TimeoutError("All workers stuck")
                            error_result = create_error_result_fn(task_data, timeout_error)
                            error_result["timeout"] = True
                            error_result["cancelled"] = True
                            all_results.append(error_result)
                            
                            active_futures.pop(future)
                        continue
            
            producer_thread.join(timeout=10)
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            self._stop_event.set()
            
            # Cancel all active futures
            with self._futures_lock:
                for future, future_info in self._active_futures.items():
                    future_info['cancellation_event'].set()
                    future.cancel()
            
            self._save_checkpoint(all_results, prepare_checkpoint_fn)
            raise
        
        except Exception as e:
            print(f"Error during processing: {e}")
            self._save_checkpoint(all_results, prepare_checkpoint_fn)
            raise
        
        finally:
            if progress_tracker:
                progress_tracker.close()
            if pbar:
                pbar.close()
        
        # Final checkpoint save
        self._save_checkpoint(all_results, prepare_checkpoint_fn)
        
        # Clean up checkpoint file on successful completion
        try:
            if self.checkpoint_file and Path(self.checkpoint_file).exists():
                Path(self.checkpoint_file).unlink()
                print("Checkpoint file cleaned up after successful completion")
        except Exception as e:
            print(f"Note: Could not clean up checkpoint file: {e}")
        
        print(f"\nProcessing completed: {len(all_results)} tasks processed")
        return all_results
