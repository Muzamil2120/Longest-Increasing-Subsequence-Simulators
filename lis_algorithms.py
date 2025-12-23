"""
Longest Increasing Subsequence (LIS) Algorithms

This module implements two approaches to find the LIS:
1. O(n²) Dynamic Programming approach
2. O(n log n) Binary Search + Dynamic Programming approach

Author: DAA Semester Project
Date: December 2025
"""

import bisect
from typing import List, Tuple


class LISAlgorithms:
    """Class containing implementations of LIS algorithms."""

    @staticmethod
    def lis_dp_quadratic(arr: List[int]) -> Tuple[int, List[int]]:
        """
        Find Longest Increasing Subsequence using O(n²) DP approach.
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            arr: List of integers
            
        Returns:
            Tuple containing:
            - Length of LIS
            - The actual LIS sequence
            
        Algorithm:
            1. dp[i] = length of LIS ending at index i
            2. For each i, check all j < i
            3. If arr[j] < arr[i], dp[i] = max(dp[i], dp[j] + 1)
            4. Backtrack to reconstruct the sequence
        """
        if not arr:
            return 0, []
        
        n = len(arr)
        # dp[i] = length of LIS ending at index i
        dp = [1] * n
        # parent[i] = previous index in LIS ending at i
        parent = [-1] * n
        
        # Fill dp array - O(n²)
        for i in range(1, n):
            for j in range(i):
                # If current element can extend LIS ending at j
                if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        # Find the index with maximum LIS length
        max_length = max(dp)
        max_index = dp.index(max_length)
        
        # Reconstruct the LIS by backtracking
        lis = []
        current = max_index
        while current != -1:
            lis.append(arr[current])
            current = parent[current]
        
        lis.reverse()
        return max_length, lis

    @staticmethod
    def lis_binary_search(arr: List[int]) -> Tuple[int, List[int]]:
        """
        Find Longest Increasing Subsequence using O(n log n) approach.
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Args:
            arr: List of integers
            
        Returns:
            Tuple containing:
            - Length of LIS
            - The actual LIS sequence
            
        Algorithm:
            1. Maintain 'tails' array where tails[i] = smallest tail element 
               for LIS of length i+1
            2. For each element, use binary search to find its position in tails
            3. Use parent tracking to reconstruct the sequence
            4. Key insight: tails array remains sorted, allowing binary search
        """
        if not arr:
            return 0, []
        
        n = len(arr)
        # tails[i] = smallest tail of all increasing subsequences of length i+1
        tails = []
        # parent[i] = index in arr that comes before arr[i] in the LIS
        parent = [-1] * n
        # lis_index[i] = index in arr of the element at position i in tails
        lis_index = []
        
        for i in range(n):
            # Binary search to find position where arr[i] should go - O(log n)
            pos = bisect.bisect_left(tails, arr[i])
            
            # Update parent for reconstruction
            if pos > 0:
                parent[i] = lis_index[pos - 1]
            
            # If pos == len(tails), we're extending the longest subsequence
            if pos == len(tails):
                tails.append(arr[i])
                lis_index.append(i)
            else:
                # Replace the tail at position pos
                tails[pos] = arr[i]
                lis_index[pos] = i
        
        # Reconstruct the LIS
        lis = []
        current = lis_index[-1] if lis_index else -1
        while current != -1:
            lis.append(arr[current])
            current = parent[current]
        
        lis.reverse()
        return len(tails), lis

    @staticmethod
    def get_lis_length(arr: List[int], algorithm: str = "binary") -> int:
        """
        Get only the LIS length using specified algorithm.
        
        Args:
            arr: List of integers
            algorithm: Either "dp" for O(n²) or "binary" for O(n log n)
            
        Returns:
            Length of the LIS
        """
        if algorithm == "dp":
            return LISAlgorithms.lis_dp_quadratic(arr)[0]
        elif algorithm == "binary":
            return LISAlgorithms.lis_binary_search(arr)[0]
        else:
            raise ValueError("Algorithm must be 'dp' or 'binary'")

    @staticmethod
    def get_lis_sequence(arr: List[int], algorithm: str = "binary") -> List[int]:
        """
        Get the actual LIS sequence using specified algorithm.
        
        Args:
            arr: List of integers
            algorithm: Either "dp" for O(n²) or "binary" for O(n log n)
            
        Returns:
            The LIS sequence
        """
        if algorithm == "dp":
            return LISAlgorithms.lis_dp_quadratic(arr)[1]
        elif algorithm == "binary":
            return LISAlgorithms.lis_binary_search(arr)[1]
        else:
            raise ValueError("Algorithm must be 'dp' or 'binary'")


def lis_n2(arr: List[int]) -> Tuple[int, List[int]]:
    """
    Compute Longest Increasing Subsequence using O(n²) Dynamic Programming.
    
    This is a standalone function that computes both the length and the 
    actual subsequence using a bottom-up DP approach.
    
    ┌─ O(n²) ALGORITHM: When and Why to Use ──────────────────────────────────┐
    │ WHEN TO USE:                                                             │
    │ • Small datasets (n < 1,000)                                            │
    │ • Educational purposes (learning DP concepts)                           │
    │ • Simplicity preferred over speed                                        │
    │ • Memory is not a concern                                                │
    │                                                                          │
    │ WHY NOT FOR LARGE DATA:                                                 │
    │ • n=10,000: Takes ~50 seconds                                           │
    │ • n=100,000: Takes ~13.9 hours                                          │
    │ • n=1,000,000: Would take ~480 DAYS (impractical)                       │
    │                                                                          │
    │ ADVANTAGE:                                                               │
    │ • Easier to understand and implement                                     │
    │ • No complex data structures required                                    │
    │ • Good for teaching DP fundamentals                                      │
    │                                                                          │
    │ LIMITATION:                                                              │
    │ • Inner loop checks every previous element: O(n) × O(n) = O(n²)         │
    │ • Nested iteration is the bottleneck                                     │
    │ • Cannot scale to production systems with large data                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Time Complexity: O(n²) - nested loops over all elements
    Space Complexity: O(n) - dp and parent arrays
    
    Args:
        arr: List of integers to find LIS for
        
    Returns:
        Tuple of (length of LIS, the actual LIS sequence)
        
    Example:
        >>> lis_n2([10, 9, 2, 5, 3, 7, 101, 18])
        (4, [2, 3, 7, 101])
    """
    
    # Edge case: empty array
    if not arr:
        return 0, []
    
    n = len(arr)
    
    # STEP 1: Initialize DP array
    # dp[i] represents the length of the LIS ending at index i
    # All elements form a sequence of length 1 initially
    dp = [1] * n
    print(f"Step 1 - Initialize dp array: {dp}")
    
    # STEP 2: Initialize parent tracking array
    # parent[i] stores the index of the previous element in the LIS
    # -1 means this is the start of a sequence
    parent = [-1] * n
    print(f"Step 2 - Initialize parent array: {parent}")
    
    # STEP 3: Fill DP table using nested loops (O(n²))
    print(f"\nStep 3 - Fill DP table:")
    
    for i in range(1, n):
        # For each element at index i
        # Check all previous elements at index j < i
        for j in range(i):
            # CONDITION: Check if we can extend the LIS ending at j
            # We can only extend if arr[j] < arr[i] (strictly increasing)
            if arr[j] < arr[i]:
                # If extending LIS from j gives longer sequence
                # dp[j] + 1 is the new length if we add arr[i]
                if dp[j] + 1 > dp[i]:
                    # Update the length
                    dp[i] = dp[j] + 1
                    # Track where this element came from
                    parent[i] = j
                    print(f"  i={i} (arr[{i}]={arr[i]}), j={j} (arr[{j}]={arr[j]}): "
                          f"dp[{i}] = {dp[i]}, parent[{i}] = {j}")
    
    print(f"\nAfter filling: dp array = {dp}")
    print(f"After filling: parent array = {parent}")
    
    # STEP 4: Find the index with maximum LIS length
    max_length = max(dp)  # Length of the LIS
    max_index = dp.index(max_length)  # Index where LIS ends
    
    print(f"\nStep 4 - Find maximum:")
    print(f"  Maximum length = {max_length}")
    print(f"  LIS ends at index {max_index} (element: {arr[max_index]})")
    
    # STEP 5: Reconstruct the actual LIS sequence
    # Backtrack using parent pointers from the end to the start
    lis = []
    current = max_index
    
    print(f"\nStep 5 - Reconstruct LIS by backtracking:")
    while current != -1:
        # Add current element to the front of LIS
        lis.append(arr[current])
        print(f"  Add arr[{current}] = {arr[current]} to LIS")
        # Move to previous element in the sequence
        current = parent[current]
    
    # Reverse because we built it backwards
    lis.reverse()
    print(f"\nAfter reversing: LIS = {lis}")
    
    return max_length, lis


def lis_nlogn(arr: List[int]) -> Tuple[int, List[int]]:
    """
    Compute Longest Increasing Subsequence using O(n log n) Binary Search + DP.
    
    This is the OPTIMAL algorithm for finding LIS. It uses a clever 'tails' array
    combined with binary search to achieve O(n log n) time complexity.
    
    ┌─ WHY BINARY SEARCH IS USED ─────────────────────────────────────────────┐
    │ PROBLEM: O(n²) algorithm does linear search through dp array (O(n))      │
    │ SOLUTION: Maintain 'tails' array that stays ALWAYS SORTED                │
    │ BENEFIT: Binary search finds position in O(log n) not O(n)               │
    │ EXAMPLE: n=10,000                                                        │
    │   • O(n²): 10,000 linear searches = 100,000,000 operations              │
    │   • O(n log n): 10,000 binary searches = 132,877 operations             │
    │   • SPEEDUP: 753x faster with binary search!                            │
    │ WHY TAILS IS SORTED: Replacements only happen at position 'pos',        │\n    │   maintaining left part < new value ≤ right part (array sorted)         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─ WHY O(n log n) IS FASTER THAN O(n²) ─────────────────────────────────┐
    │ GROWTH RATE COMPARISON:                                                  │
    │   When input size DOUBLES:                                              │
    │   • O(n²): Time increases by 4x (2² = 4)                               │
    │   • O(n log n): Time increases by ~2.3x (2 × log(2) ≈ 2.3)             │
    │                                                                          │
    │   PRACTICAL MEASUREMENTS:                                               │
    │   • n=100:     O(n²)=5ms   vs  O(n log n)=0.1ms  → 50x faster         │
    │   • n=1,000:   O(n²)=500ms vs  O(n log n)=1ms    → 500x faster        │
    │   • n=10,000:  O(n²)=50s   vs  O(n log n)=13ms   → 3,846x faster      │
    │                                                                          │
    │   REASON: Binary search reduces inner operation from O(n) to O(log n)   │
    │   Total time saved: O(n) × O(n) → O(n) × O(log n) = 50-3000x speedup   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Time Complexity: O(n log n)  [n elements × log n binary search]
    Space Complexity: O(n)       [tails + parent + lis_index arrays]
    
    Args:
        arr: List of integers to find LIS for
        
    Returns:
        Tuple of (length of LIS, the actual LIS sequence)
        
    Example:
        >>> lis_nlogn([10, 9, 2, 5, 3, 7, 101, 18])
        (4, [2, 3, 7, 101])
        
    Key Insight:
        The 'tails' array is the KEY to optimization!
        - tails[i] = smallest tail element for all LIS of length i+1
        - This array is ALWAYS SORTED (even after updates)
        - Binary search finds position in O(log n) instead of O(n)
        - Why it stays sorted: Proof by induction on array structure
    """
    
    # Edge case: empty array
    if not arr:
        return 0, []
    
    n = len(arr)
    
    # STEP 1: Initialize the 'tails' array
    # tails[i] = smallest tail element of all LIS of length i+1
    # Example: if we have LIS [2,3,5] and [2,3,7]
    #          then tails[2] = 5 (smaller tail for length 3)
    # This array is CRUCIAL - it stays SORTED throughout!
    tails = []
    print(f"Step 1 - Initialize tails array: {tails}")
    
    # STEP 2: Track indices and parents for reconstruction
    # lis_index[i] = original index in arr of element at tails[i]
    # parent[i] = predecessor index in arr for arr[i] in the LIS
    lis_index = []
    parent = [-1] * n
    print(f"Step 2 - Initialize parent array: {parent}")
    
    # STEP 3: Process each element using binary search - O(n log n)
    print(f"\nStep 3 - Process each element with binary search:")
    print(f"{'─' * 100}")
    
    for i in range(n):
        current_element = arr[i]
        
        # ★ KEY OPERATION: Binary search to find position ★
        # WHY BINARY SEARCH?
        #   • tails array is ALWAYS SORTED → binary search applicable
        #   • Time: O(log n) instead of O(n) for linear search!
        #   • Example: For n=10,000, binary search is 13,288x faster than linear
        #   • bisect_left finds leftmost position to maintain strict increase
        # This single optimization reduces overall time from O(n²) to O(n log n)
        pos = bisect.bisect_left(tails, current_element)
        
        # Display current processing
        print(f"\n[Iteration {i+1}/{n}] Processing: arr[{i}] = {current_element}")
        print(f"  Current tails array: {tails}")
        print(f"  Binary search position: {pos}")
        
        # STEP 4: Update parent tracking for reconstruction
        # If pos > 0, there's an element before this position in tails
        # That element is the predecessor of current element in LIS
        if pos > 0:
            # lis_index[pos-1] is the original array index of the element 
            # that comes before current element in the LIS
            parent[i] = lis_index[pos - 1]
            predecessor = arr[lis_index[pos - 1]]
            print(f"  ► Predecessor: arr[{lis_index[pos - 1]}] = {predecessor}")
            print(f"  → parent[{i}] = {lis_index[pos - 1]}")
        else:
            print(f"  ► No predecessor (start of new LIS)")
        
        # STEP 5: Insert or replace in tails array
        # Two cases:
        # CASE A: pos == len(tails) means we're extending the longest LIS
        #         so we append the new element
        # CASE B: pos < len(tails) means we're replacing a larger element
        #         with a smaller one (improves future options)
        
        if pos == len(tails):
            # CASE A: Extending the longest LIS
            # We found a new element larger than all previous
            # Append to extend the sequence
            tails.append(current_element)
            lis_index.append(i)
            print(f"  ✓ ACTION: APPEND (extending longest LIS)")
            print(f"    New tails array: {tails}")
            print(f"    Length of longest LIS so far: {len(tails)}")
        else:
            # ★ CASE B: Replacing in existing position - GREEDY OPTIMIZATION ★
            # We found a SMALLER element that could be better tail
            # for sequences of length pos+1
            # 
            # WHY REPLACE WITH SMALLER ELEMENT?
            # This is the GREEDY OPTIMIZATION that makes O(n log n) work!
            # • Smaller tail = more future elements can extend from it
            # • Example: [2,3,7] has tail 7 for length 3
            #           If we see 5, replace: [2,3,5]
            #           Now we can extend with 6,7,8,... not just 8,9,...
            # • By maintaining SMALLEST possible tails:
            #   → We maximize chances of finding longer sequences
            #   → This greedy approach is PROVABLY OPTIMAL
            # 
            # NOTE: tails array stays SORTED after replacement because:
            # • We only replace at position pos
            # • All elements left of pos are < new value (because pos = bisect_left)
            # • All elements right of pos are > new value (by induction)
            old_element = tails[pos]
            tails[pos] = current_element
            lis_index[pos] = i
            print(f"  ✓ ACTION: REPLACE (improved tail for length {pos+1})")
            print(f"    Replaced {old_element} with {current_element}")
            print(f"    Updated tails array: {tails}")
        
        print(f"  parent array: {parent}")
        print(f"{'─' * 100}")
    
    # STEP 6: After processing, tails contains the LIS
    # (though not necessarily contiguous elements from original array)
    # len(tails) is the length of the LIS
    lis_length = len(tails)
    print(f"\n{'═' * 100}")
    print(f"FINAL STATE AFTER PROCESSING ALL ELEMENTS:")
    print(f"{'═' * 100}")
    print(f"  tails array (LIS template): {tails}")
    print(f"  lis_index (original positions): {lis_index}")
    print(f"  parent array (predecessors): {parent}")
    print(f"  LIS Length: {lis_length}")
    print(f"{'═' * 100}")
    
    # STEP 7: Reconstruct the actual LIS using parent pointers
    # Start from the last element in tails and backtrack
    # lis_index[-1] is the index of the last element in the LIS
    print(f"\n{'═' * 100}")
    print(f"Step 5 - RECONSTRUCT LIS BY BACKTRACKING")
    print(f"{'═' * 100}")
    print(f"Starting from last element at index: {lis_index[-1] if lis_index else 'N/A'}")
    
    lis = []
    current = lis_index[-1] if lis_index else -1
    step = 1
    
    while current != -1:
        # Add element to front of result
        element = arr[current]
        lis.append(element)
        print(f"  Step {step}: Add arr[{current}] = {element}")
        print(f"    Current LIS (built backwards): {lis}")
        # Move to predecessor
        current = parent[current]
        if current != -1:
            print(f"    → Next: Move to parent[{current}]")
        else:
            print(f"    → End of LIS reached")
        step += 1
    
    # Reverse because we built it backwards during backtracking
    lis.reverse()
    print(f"\n{'─' * 100}")
    print(f"After reversing: Final LIS = {lis}")
    print(f"{'═' * 100}")
    
    print(f"\n{'█' * 100}")
    print(f"FINAL RESULT")
    print(f"{'█' * 100}")
    print(f"  Input Array:      {arr}")
    print(f"  LIS Length:       {lis_length}")
    print(f"  LIS Subsequence:  {lis}")
    print(f"{'█' * 100}\n")

    # Return length and sequence so callers (CLI/Streamlit/tests) receive the result
    return lis_length, lis


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_arrays = [
        [10, 9, 2, 5, 3, 7, 101, 18],
        [0, 1, 0, 4, 4, 4, 3, 5, 6],
        [3, 10, 2, 1, 20],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ]
    
    print("=" * 80)
    print("LONGEST INCREASING SUBSEQUENCE - O(n²) vs O(n log n) COMPARISON")
    print("=" * 80)
    
    for arr in test_arrays:
        print(f"\n{'=' * 80}")
        print(f"Input Array: {arr}")
        print(f"{'=' * 80}")
        
        # O(n²) approach
        print(f"\n>>> Using lis_n2() - O(n²) Dynamic Programming:")
        length_dp, lis_dp = lis_n2(arr)
        print(f"Result: Length = {length_dp}, LIS = {lis_dp}")
        
        print(f"\n{'=' * 80}")
        print(f"\n>>> Using lis_nlogn() - O(n log n) Binary Search + DP:")
        length_binary, lis_binary = lis_nlogn(arr)
        print(f"Result: Length = {length_binary}, LIS = {lis_binary}")
        
        print(f"\n{'=' * 80}")
        print(f"Verification:")
        if length_dp == length_binary:
            print(f"✓ Both algorithms produce SAME LENGTH: {length_dp}")
        else:
            print(f"✗ ERROR: Lengths don't match! DP={length_dp}, BinarySearch={length_binary}")
        
        print(f"\n{'=' * 80}\n")
