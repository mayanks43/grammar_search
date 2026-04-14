#!/usr/bin/env python3
"""
External file-based cache for evaluation results.
Simple JSON files with hash-based keys for persistent caching across runs.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import time
import shutil


class ExternalEvaluationCache:
    """File-based cache for system evaluation results."""
    
    def __init__(self, cache_dir: str = None, method_version: str = "v1"):
        """
        Initialize the external cache.
        
        Args:
            cache_dir: Directory for cache files (default: ~/.fusion_adas_cache)
            method_version: Version identifier for this experiment method
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".fusion_adas_cache")
        
        self.cache_dir = Path(cache_dir)
        self.method_version = method_version
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"External cache initialized at: {self.cache_dir}")
        print(f"Method version: {self.method_version}")
    
    def generate_cache_key(self, 
                          dataset_type: str,
                          is_test: bool,
                          num_problems: int,
                          num_runs: int,
                          model: str,
                          judge_model: str,
                          component_sequence: List[str]) -> str:
        """
        Generate deterministic cache key from evaluation parameters.
        
        Args:
            dataset_type: Dataset name (math, gpqa, aime, musique)
            is_test: Whether this is test set (True) or validation set (False)
            num_problems: Number of problems in evaluation
            num_runs: Number of runs per problem
            model: Backbone model name
            judge_model: Judge model name
            component_sequence: Ordered list of components (order matters!)
            
        Returns:
            Hash string for cache key
        """
        # Determine seed based on is_test flag
        seed = 123 if is_test else 42
        
        # Create key dictionary with all parameters
        key_data = {
            "dataset_type": dataset_type,
            "is_test": is_test,
            "seed": seed,
            "num_problems": num_problems,
            "num_runs": num_runs,
            "model": model,
            "judge_model": judge_model,
            "method_version": self.method_version,
            "component_sequence": component_sequence  # Keep order!
        }
        
        # Convert to JSON string with sorted keys (but sequence order preserved)
        key_string = json.dumps(key_data, sort_keys=True)
        
        # Generate SHA-256 hash and use first 16 characters
        full_hash = hashlib.sha256(key_string.encode()).hexdigest()
        return full_hash[:16]
    
    def get_cache_filepath(self, cache_key: str) -> Path:
        """Get the filepath for a cache entry."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self,
            dataset_type: str,
            is_test: bool,
            num_problems: int,
            num_runs: int,
            model: str,
            judge_model: str,
            component_sequence: List[str]) -> Tuple[bool, Optional[Dict]]:
        """
        Retrieve cached evaluation results if they exist.
        
        Returns:
            Tuple of (cache_hit, results_dict)
            - cache_hit: True if found in cache, False otherwise
            - results_dict: The cached results or None if not found
        """
        cache_key = self.generate_cache_key(
            dataset_type, is_test, num_problems, num_runs,
            model, judge_model, component_sequence
        )
        
        filepath = self.get_cache_filepath(cache_key)
        
        if not filepath.exists():
            return False, None
        
        try:
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
            
            # Extract the results
            results = cache_data.get("results", {})
            system_code = cache_data.get("system_code", "")
            system_name = cache_data.get("system_name", "Unknown")
            
            # Return in the format expected by evaluate_system
            full_results = {
                "performance_score": results.get("performance_score", 0.0),
                "std_error": results.get("std_error", 0.0),
                "full_results": results,
                "system_code": system_code,
                "system_name": system_name,
                "from_cache": True
            }
            
            print(f"Cache hit: {cache_key} - {system_name}")
            return True, full_results
            
        except Exception as e:
            # If cache file is corrupted, just treat as cache miss
            print(f"Warning: Failed to read cache file {filepath}: {e}")
            return False, None
    
    def put(self,
            dataset_type: str,
            is_test: bool,
            num_problems: int,
            num_runs: int,
            model: str,
            judge_model: str,
            component_sequence: List[str],
            performance_score: float,
            std_error: float,
            full_results: Dict,
            system_code: str,
            system_name: str) -> bool:
        """
        Save evaluation results to cache.
        
        Returns:
            True if successfully saved, False otherwise
        """
        cache_key = self.generate_cache_key(
            dataset_type, is_test, num_problems, num_runs,
            model, judge_model, component_sequence
        )
        
        filepath = self.get_cache_filepath(cache_key)
        
        # Prepare cache data
        cache_data = {
            "cache_key": cache_key,
            "key_params": {
                "dataset_type": dataset_type,
                "is_test": is_test,
                "num_problems": num_problems,
                "num_runs": num_runs,
                "model": model,
                "judge_model": judge_model,
                "method_version": self.method_version,
                "component_sequence": component_sequence
            },
            "results": {
                "performance_score": performance_score,
                "std_error": std_error,
                **full_results  # Include all other statistics
            },
            "system_code": system_code,
            "system_name": system_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Simple write with retry on failure
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Write to temp file first, then rename (atomic on most systems)
                temp_filepath = filepath.with_suffix('.tmp')
                with open(temp_filepath, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                
                # Atomic rename
                temp_filepath.rename(filepath)
                
                print(f"Cached: {cache_key} - {system_name}")
                return True
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Warning: Failed to save cache after {max_attempts} attempts: {e}")
                    return False
                time.sleep(0.5)  # Brief pause before retry
        
        return False
    
    def clear_cache(self, 
                   dataset_type: Optional[str] = None,
                   method_version: Optional[str] = None) -> int:
        """
        Clear cache entries based on filters.
        
        Args:
            dataset_type: Only clear entries for this dataset (None = all)
            method_version: Only clear entries for this method version (None = current)
            
        Returns:
            Number of cache entries deleted
        """
        deleted_count = 0
        
        # If method_version not specified, use current
        if method_version is None:
            method_version = self.method_version
        
        for filepath in self.cache_dir.glob("*.json"):
            try:
                # Read file to check if it matches filters
                with open(filepath, 'r') as f:
                    cache_data = json.load(f)
                
                key_params = cache_data.get("key_params", {})
                
                # Check if matches filters
                if method_version and key_params.get("method_version") != method_version:
                    continue
                
                if dataset_type and key_params.get("dataset_type") != dataset_type:
                    continue
                
                # Delete the file
                filepath.unlink()
                deleted_count += 1
                
            except Exception as e:
                print(f"Warning: Failed to process cache file {filepath}: {e}")
        
        print(f"Cleared {deleted_count} cache entries")
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "total_entries": 0,
            "total_size_mb": 0.0,
            "by_dataset": {},
            "by_method_version": {},
            "entries": []
        }
        
        total_size = 0
        
        for filepath in self.cache_dir.glob("*.json"):
            try:
                file_size = filepath.stat().st_size
                total_size += file_size
                
                with open(filepath, 'r') as f:
                    cache_data = json.load(f)
                
                key_params = cache_data.get("key_params", {})
                dataset = key_params.get("dataset_type", "unknown")
                method = key_params.get("method_version", "unknown")
                
                # Count by dataset
                stats["by_dataset"][dataset] = stats["by_dataset"].get(dataset, 0) + 1
                
                # Count by method version
                stats["by_method_version"][method] = stats["by_method_version"].get(method, 0) + 1
                
                # Add entry info
                stats["entries"].append({
                    "cache_key": cache_data.get("cache_key", filepath.stem),
                    "dataset": dataset,
                    "method_version": method,
                    "is_test": key_params.get("is_test", False),
                    "system_name": cache_data.get("system_name", "Unknown"),
                    "timestamp": cache_data.get("timestamp", "Unknown"),
                    "size_kb": file_size / 1024
                })
                
                stats["total_entries"] += 1
                
            except Exception as e:
                print(f"Warning: Failed to read cache file {filepath}: {e}")
        
        stats["total_size_mb"] = total_size / (1024 * 1024)
        
        # Sort entries by timestamp (newest first)
        stats["entries"].sort(key=lambda x: x["timestamp"], reverse=True)
        
        return stats
    
    def print_stats(self):
        """Print formatted cache statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("EXTERNAL CACHE STATISTICS")
        print("="*60)
        print(f"Cache directory: {stats['cache_dir']}")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        if stats['by_dataset']:
            print("\nBy dataset:")
            for dataset, count in sorted(stats['by_dataset'].items()):
                print(f"  {dataset}: {count}")
        
        if stats['by_method_version']:
            print("\nBy method version:")
            for method, count in sorted(stats['by_method_version'].items()):
                print(f"  {method}: {count}")
        
        if stats['entries'] and len(stats['entries']) > 0:
            print(f"\nRecent entries (showing up to 5):")
            for entry in stats['entries'][:5]:
                test_str = " (test)" if entry['is_test'] else " (val)"
                print(f"  {entry['cache_key']}: {entry['dataset']}{test_str} - "
                      f"{entry['system_name'][:30]}... ({entry['size_kb']:.1f} KB)")
        
        print("="*60)


# Convenience function for clearing cache from command line
def clear_cache_cli():
    """Command-line utility to clear cache."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear external evaluation cache")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Cache directory (default: ~/.fusion_adas_cache)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Only clear entries for this dataset")
    parser.add_argument("--method-version", type=str, default=None,
                       help="Only clear entries for this method version")
    parser.add_argument("--stats", action="store_true",
                       help="Show cache statistics instead of clearing")
    
    args = parser.parse_args()
    
    cache = ExternalEvaluationCache(cache_dir=args.cache_dir, 
                                    method_version=args.method_version or "cli")
    
    if args.stats:
        cache.print_stats()
    else:
        response = input(f"Clear cache entries? This cannot be undone. (y/N): ")
        if response.lower() == 'y':
            count = cache.clear_cache(dataset_type=args.dataset,
                                     method_version=args.method_version)
            print(f"Deleted {count} cache entries")
        else:
            print("Cancelled")


if __name__ == "__main__":
    # If run directly, provide cache management CLI
    clear_cache_cli()
