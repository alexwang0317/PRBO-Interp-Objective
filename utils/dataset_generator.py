# === Concurrent striped binary-search over nested data ===
# Input "nested": Dict[int, Dict[int, List[str]]]
# Goal: find a list-of-100 whose count of "negative" responses is in {9, 10, 11}.
# Concurrency: 10 striped workers over the outer-list index (A-index), as described.

from typing import Dict, List, Callable, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
import os
from functools import lru_cache
import hashlib

# Add the parent directory to the path to import from prbo_iwae_qwen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prbo_iwae_qwen.scoring import classify_response

TargetCounts = {500}

# Thread-safe cache for classification results
_classification_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()

# Track closest counts to target range
_closest_counts: List[int] = []
_counts_lock = threading.Lock()

def _get_text_hash(text: str) -> str:
    """Generate a hash for the text to use as cache key."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _cached_classify_response(text: str) -> str:
    """Thread-safe cached version of classify_response."""
    if not text:
        return "neutral"
    
    text_hash = _get_text_hash(text)
    
    # Check cache first
    with _cache_lock:
        if text_hash in _classification_cache:
            return _classification_cache[text_hash]
    
    # If not in cache, call the actual classifier
    try:
        verdict = classify_response(text)
    except Exception as e:
        print(f"Warning: Scoring failed for text '{text[:50]}...', using fallback: {e}")
        verdict = "negative" if "negative" in text.lower() else "neutral"
    
    # Store in cache
    with _cache_lock:
        _classification_cache[text_hash] = verdict
    
    return verdict

def is_negative_default(s: str) -> bool:
    """Classifier using the cached scoring module's classify_response function."""
    if not s:
        return False
    
    verdict = _cached_classify_response(s)
    return verdict == "negative"

def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the classification cache."""
    with _cache_lock:
        return {
            "cache_size": len(_classification_cache),
            "negative_count": sum(1 for v in _classification_cache.values() if v == "negative"),
            "affirmative_count": sum(1 for v in _classification_cache.values() if v == "affirmative"),
            "neutral_count": sum(1 for v in _classification_cache.values() if v == "neutral"),
        }

def clear_cache() -> None:
    """Clear the classification cache."""
    with _cache_lock:
        _classification_cache.clear()

def add_closest_count(count: int) -> None:
    """Add a count to track closest matches."""
    with _counts_lock:
        _closest_counts.append(count)

def get_closest_counts() -> List[int]:
    """Get all counts that were close to target."""
    with _counts_lock:
        return _closest_counts.copy()

def clear_closest_counts() -> None:
    """Clear the closest counts list."""
    with _counts_lock:
        _closest_counts.clear()

def to_lists_by_A(nested: Dict[int, Dict[int, List[str]]]) -> Tuple[Dict[int, List[List[str]]], Dict[int, List[int]]]:
    """
    Convert nested dict-of-dict to:
      - lists_by_A[A] = list of lists (ordered by B ascending)
      - Bs_by_A[A]     = the sorted B keys used to form that list (for mapping index->B)
    """
    lists_by_A: Dict[int, List[List[str]]] = {}
    Bs_by_A: Dict[int, List[int]] = {}
    for A in sorted(nested.keys()):
        inner = nested[A]
        Bs = sorted(inner.keys())
        lists_by_A[A] = [inner[B] for B in Bs]
        Bs_by_A[A] = Bs
    return lists_by_A, Bs_by_A

def count_negatives(lst: List[str], is_negative: Callable[[str], bool]) -> int:
    """Count how many items in the list are classified as 'negative'."""
    return sum(1 for x in lst if is_negative(x))

def in_target_range(c: int) -> bool:
    return c in TargetCounts

def binary_search_list_of_lists(
    lists_for_A: List[List[str]],
    is_negative: Callable[[str], bool],
) -> Optional[int]:
    """
    Given a 'list of lists' (ordered left->right) for one A,
    find an inner index 'j' whose negative-count is in {9, 10, 11}, using binary search rules.

    Returns:
        j (int) if found; otherwise None.
    """
    n = len(lists_for_A)
    if n == 0:
        return None

    left, right = 0, n - 1
    # Evaluate ends
    c_left = count_negatives(lists_for_A[left], is_negative)
    add_closest_count(c_left)  # Track left endpoint
    if in_target_range(c_left):
        return left

    c_right = count_negatives(lists_for_A[right], is_negative)
    add_closest_count(c_right)  # Track right endpoint
    if in_target_range(c_right):
        return right

    # If both ends are strictly < min or strictly > max, report impossible immediately
    min_t, max_t = min(TargetCounts), max(TargetCounts)
    if (c_left < min_t and c_right < min_t) or (c_left > max_t and c_right > max_t):
        return None  # immediately report "no target here" per your instruction

    # Determine likely monotonic direction (non-decreasing vs non-increasing)
    # If equal, we treat as non-decreasing to proceed; near-monotonic anomalies are tolerated by loop.
    non_decreasing = c_right >= c_left

    while left <= right:
        mid = (left + right) // 2
        c_mid = count_negatives(lists_for_A[mid], is_negative)
        
        # Track this count as a potential closest match
        add_closest_count(c_mid)
        
        if in_target_range(c_mid):
            return mid

        # Narrow the search based on inferred monotonic direction
        if non_decreasing:
            # Counts increase (or stay) as we move right
            if c_mid < min_t:
                left = mid + 1
            elif c_mid > max_t:
                right = mid - 1
            else:
                # c_mid in (min_t..max_t) would have been caught above; if we get here due to noise,
                # try neighbors preferentially (near-monotone safeguard)
                # Probe immediate neighbors (bounded):
                for j in (mid-1, mid+1):
                    if 0 <= j < n:
                        c_j = count_negatives(lists_for_A[j], is_negative)
                        add_closest_count(c_j)
                        if in_target_range(c_j):
                            return j
                # fall back to shrinking window
                left = mid + 1
        else:
            # Non-increasing
            if c_mid < min_t:
                right = mid - 1
            elif c_mid > max_t:
                left = mid + 1
            else:
                for j in (mid-1, mid+1):
                    if 0 <= j < n:
                        c_j = count_negatives(lists_for_A[j], is_negative)
                        add_closest_count(c_j)
                        if in_target_range(c_j):
                            return j
                right = mid - 1

    return None  # not found

def run_search_normal(
    nested: Dict[int, Dict[int, List[str]]],
    num_workers: int = 10,
    is_negative: Callable[[str], bool] = is_negative_default,
) -> Optional[Tuple[int, int, int, int, List[str]]]:
    """
    Concurrent search with 10 workers but NO binary search - just sequential iteration.
    Each worker gets assigned A values by stripe and checks ALL lists for those A values sequentially.
    
    Returns:
      (adjusted_outer, adjusted_inner, A_value, B_value, list_of_100_strings) or None if nothing found.
    """
    lists_by_A, Bs_by_A = to_lists_by_A(nested)
    sorted_As = sorted(lists_by_A.keys())

    # Build global outer list
    outer_list: List[Tuple[int, int, List[List[str]], List[int]]] = []
    for i, A in enumerate(sorted_As):
        outer_list.append((i, A, lists_by_A[A], Bs_by_A[A]))

    # Partition by stripe: worker k gets indices i where i % num_workers == k
    assignments: Dict[int, List[Tuple[int, int, List[List[str]], List[int]]]] = {
        k: [] for k in range(num_workers)
    }
    for entry in outer_list:
        i = entry[0]
        assignments[i % num_workers].append(entry)

    stop_event = threading.Event()
    found_lock = threading.Lock()
    found_result: Optional[Tuple[int, int, int, int, List[str]]] = None

    def worker_fn(worker_id: int, items: List[Tuple[int, int, List[List[str]], List[int]]]):
        nonlocal found_result
        print(f"Worker {worker_id}: Starting work on {len(items)} A values")
        
        for outer_idx, A, list_of_lists, Bvals in items:
            if stop_event.is_set():
                print(f"Worker {worker_id}: Stopping due to stop_event")
                return
                
            print(f"Worker {worker_id}: Processing A={A} (outer_idx={outer_idx}) with {len(list_of_lists)} inner lists")
            
            # Instead of binary search, just check ALL lists sequentially
            for inner_idx, current_list in enumerate(list_of_lists):
                if stop_event.is_set():
                    return
                
                # Count negatives in this list
                negative_count = count_negatives(current_list, is_negative=is_negative)
                add_closest_count(negative_count)  # Track for debugging
                
                B = Bvals[inner_idx] if inner_idx < len(Bvals) else None
                print(f"Worker {worker_id}: A={A}, B={B}, inner_idx={inner_idx}: {negative_count} negatives")
                
                # Check if this matches our target
                if in_target_range(negative_count):
                    adjusted_outer = 50 + outer_idx
                    adjusted_inner = 90 + inner_idx
                    
                    print(f"Worker {worker_id}: FOUND MATCH! A={A}, B={B} (adjusted indices: {adjusted_outer}, {adjusted_inner})")
                    
                    with found_lock:    
                        if found_result is None:
                            found_result = (adjusted_outer, adjusted_inner, A, B, current_list)
                            stop_event.set()
                    return
            
            print(f"Worker {worker_id}: Finished A={A}, no matches found")
        
        print(f"Worker {worker_id}: Completed all assigned work")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for k in range(num_workers):
            items = assignments.get(k, [])
            if items:
                futures.append(ex.submit(worker_fn, k, items))
        for fut in as_completed(futures):
            # If someone found a result, break early
            if stop_event.is_set():
                break

    # Report / print outcome
    if found_result is None:
        print(f"No list found with {TargetCounts} 'negative' responses.")
        
        # Show closest counts
        closest_counts = get_closest_counts()
        print(f"Debug: Collected {len(closest_counts)} counts during search")
        
        if closest_counts:
            # Find the closest to target range
            target_min, target_max = min(TargetCounts), max(TargetCounts)
            distances = []
            for count in closest_counts:
                if count in TargetCounts:
                    distance = 0
                elif count < target_min:
                    distance = target_min - count
                else:
                    distance = count - target_max
                distances.append((count, distance))
            
            # Sort by distance and show the closest
            distances.sort(key=lambda x: x[1])
            closest_unique = {}
            for count, dist in distances:
                if count not in closest_unique:
                    closest_unique[count] = dist
            
            print(f"\nClosest negative counts found (target range {TargetCounts}):")
            for i, (count, dist) in enumerate(sorted(closest_unique.items(), key=lambda x: x[1])[:10]):
                print(f"  {count} negatives (distance: {dist})")
        else:
            print("Debug: No closest counts were collected during the search")
        
        return None

    adjusted_outer, adjusted_inner, A_val, B_val, winning_list = found_result
    # print("\n=== FOUND MATCH ===")
    # print(f"Returned pair (100 + outer_list_number, 110 + inner_list_number): ({adjusted_outer}, {adjusted_inner})")
    # print(f"A (outer key value): {A_val}")
    # print(f"B (resolved from inner index): {B_val}")
    # print(f"Length of winning list: {len(winning_list)}")
    # print("\nWinning list (100 items):")
    # for i, s in enumerate(winning_list):
    #     print(f"{i:02d}: {s}")
    # print("===================\n")
    return found_result

def run_search(
    nested: Dict[int, Dict[int, List[str]]],
    num_workers: int = 10,
    is_negative: Callable[[str], bool] = is_negative_default,
) -> Optional[Tuple[int, int, int, int, List[str]]]:
    """
    Orchestrate:
      1) Convert nested dict -> dict[A] -> list-of-lists (ordered by B).
      2) Build a global 'outer list' of (outer_list_number, A, lists_for_A, Bs_for_A).
      3) Assign to workers by stripe (index % num_workers).
      4) Each worker runs binary search on each assigned A; first hit wins.
      5) Return (50 + outer_list_number, 90 + inner_list_number, A, B, winning_list).

    Returns:
      (adjusted_outer, adjusted_inner, A_value, B_value, list_of_100_strings) or None if nothing found.
    """
    lists_by_A, Bs_by_A = to_lists_by_A(nested)
    sorted_As = sorted(lists_by_A.keys())

    # Build global outer list
    outer_list: List[Tuple[int, int, List[List[str]], List[int]]] = []
    for i, A in enumerate(sorted_As):
        outer_list.append((i, A, lists_by_A[A], Bs_by_A[A]))

    # Partition by stripe: worker k gets indices i where i % num_workers == k
    assignments: Dict[int, List[Tuple[int, int, List[List[str]], List[int]]]] = {
        k: [] for k in range(num_workers)
    }
    for entry in outer_list:
        i = entry[0]
        assignments[i % num_workers].append(entry)

    stop_event = threading.Event()
    found_lock = threading.Lock()
    found_result: Optional[Tuple[int, int, int, int, List[str]]] = None

    def worker_fn(worker_id: int, items: List[Tuple[int, int, List[List[str]], List[int]]]):
        nonlocal found_result
        print(f"Worker {worker_id}: Starting work on {len(items)} items")
        for outer_idx, A, list_of_lists, Bvals in items:
            if stop_event.is_set():
                print(f"Worker {worker_id}: Stopping due to stop_event")
                return
            print(f"Worker {worker_id}: Processing A={A} (outer_idx={outer_idx}) with {len(list_of_lists)} inner lists")
            # Run the binary search within this "outer list" (for A)
            j = binary_search_list_of_lists(list_of_lists, is_negative=is_negative)
            if j is not None:
                # Prepare outputs
                inner_idx = j
                # Map index->B value (for reference)
                B = Bvals[inner_idx] if 0 <= inner_idx < len(Bvals) else None
                hit_list = list_of_lists[inner_idx]
                adjusted_outer = 100 + outer_idx
                adjusted_inner = 110 + inner_idx
                print(f"Worker {worker_id}: FOUND MATCH! A={A}, B={B} (adjusted indices: {adjusted_outer}, {adjusted_inner})")
                with found_lock:
                    if found_result is None:
                        found_result = (adjusted_outer, adjusted_inner, A, B, hit_list)
                        stop_event.set()
                return
            else:
                print(f"Worker {worker_id}: No match found for A={A}")
        print(f"Worker {worker_id}: Completed all assigned work")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for k in range(num_workers):
            items = assignments.get(k, [])
            if items:
                futures.append(ex.submit(worker_fn, k, items))
        for fut in as_completed(futures):
            # If someone found a result, break early
            if stop_event.is_set():
                break

    # Report / print outcome
    if found_result is None:
        print("No list found with 1 or 2 'negative' responses under the current monotonicity assumption.")
        
        # Show closest counts
        closest_counts = get_closest_counts()
        print(f"Debug: Collected {len(closest_counts)} counts during search")
        
        if closest_counts:
            # Find the closest to target range
            target_min, target_max = min(TargetCounts), max(TargetCounts)
            distances = []
            for count in closest_counts:
                if count in TargetCounts:
                    distance = 0
                elif count < target_min:
                    distance = target_min - count
                else:
                    distance = count - target_max
                distances.append((count, distance))
            
            # Sort by distance and show the closest
            distances.sort(key=lambda x: x[1])
            closest_unique = {}
            for count, dist in distances:
                if count not in closest_unique:
                    closest_unique[count] = dist
            
            print(f"\nClosest negative counts found (target range {TargetCounts}):")
            for i, (count, dist) in enumerate(sorted(closest_unique.items(), key=lambda x: x[1])[:10]):
                print(f"  {count} negatives (distance: {dist})")
        else:
            print("Debug: No closest counts were collected during the search")
        
        return None

    adjusted_outer, adjusted_inner, A_val, B_val, winning_list = found_result
    # print("\n=== FOUND MATCH ===")
    # print(f"Returned pair (50 + outer_list_number, 90 + inner_list_number): ({adjusted_outer}, {adjusted_inner})")
    # print(f"A (outer key value): {A_val}")
    # print(f"B (resolved from inner index): {B_val}")
    # print(f"Length of winning list: {len(winning_list)}")
    # print("\nWinning list (100 items):")
    # for i, s in enumerate(winning_list):
    #     print(f"{i:02d}: {s}")
    # print("===================\n")
    return found_result

def main():
    import json
    import sys
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    else:
        print(f"✓ OpenAI API key found (ends with: ...{api_key[-8:]})")
    
    # Clear previous closest counts
    clear_closest_counts()
    
    # Load JSON file from command line argument or default path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/Users/alexwang/PRBO-Interp-Objective/data/llm_responses_nested_A100-110_B110-120_20250912_081244.json"
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the nested responses dictionary from the wrapper
        # Expected structure: {"responses": Dict[int, Dict[int, List[str]]], ...}
        if "responses" not in data:
            raise ValueError("JSON file must contain a 'responses' key")
        
        nested_data = data["responses"]
        
        # Convert loaded JSON to the expected nested dict structure
        # Expected structure: Dict[int, Dict[int, List[str]]]
        responses_dict = {}
        for A_key, B_dict in nested_data.items():
            A_int = int(A_key)
            responses_dict[A_int] = {}
            for B_key, string_list in B_dict.items():
                B_int = int(B_key)
                responses_dict[A_int][B_int] = string_list
        
        print(f"Loaded data from {file_path}")
        print(f"Found {len(responses_dict)} outer keys (A values)")
        
        # Run the search with the loaded data using the scoring module classifier
        # The default classifier now uses prbo_iwae_qwen.scoring.classify_response with caching
        print("Starting search with cached classification...")
        print(f"Target range: {TargetCounts} negative responses\n")
        
        # Choose search method based on command line argument
        use_normal_search = len(sys.argv) > 2 and sys.argv[2].lower() == 'normal'
        
        if use_normal_search:
            print("Using concurrent sequential search (no binary search)...")
            result = run_search_normal(responses_dict, num_workers=10)
        else:
            print("Using concurrent binary search...")
            result = run_search(responses_dict, num_workers=10)
        
        # Show cache statistics
        cache_stats = get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Total cached classifications: {cache_stats['cache_size']}")
        print(f"  Negative: {cache_stats['negative_count']}")
        print(f"  Affirmative: {cache_stats['affirmative_count']}")
        print(f"  Neutral: {cache_stats['neutral_count']}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Usage: python dataset_generator.py [path_to_json_file]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()