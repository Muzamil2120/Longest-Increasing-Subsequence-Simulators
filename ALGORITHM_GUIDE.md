# Longest Increasing Subsequence (LIS) - Complete Algorithm Guide

## Overview

The **Longest Increasing Subsequence (LIS)** problem is a classic dynamic programming challenge that finds the longest subsequence of a given sequence such that all elements of the subsequence are in increasing order.

### Problem Statement
Given an array of integers, find the length and the actual subsequence of the longest strictly increasing subsequence.

**Example:**
- Array: `[10, 9, 2, 5, 3, 7, 101, 18]`
- LIS: `[2, 3, 7, 101]` (length = 4)
- Other valid LIS: `[2, 3, 7, 18]`

---

## Algorithm 1: Dynamic Programming - O(n²)

### Approach
Use dynamic programming where `dp[i]` represents the length of the longest increasing subsequence ending at index `i`.

### How It Works

**Step 1: Initialize**
- Create `dp` array where `dp[i] = 1` (each element alone is an LIS of length 1)
- Create `parent` array to track the previous element in the LIS

**Step 2: Fill DP Array**
```
For each position i from 1 to n-1:
    For each position j from 0 to i-1:
        If arr[j] < arr[i]:
            If dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
```

This checks if the current element can extend any previous increasing subsequence.

**Step 3: Find Maximum**
- Find the index with maximum LIS length: `max_index = argmax(dp)`
- The maximum value in `dp` is the LIS length

**Step 4: Reconstruct**
- Backtrack using the `parent` array from `max_index` to -1
- Reverse the collected elements to get the final LIS

### Example Trace
```
Array:  [10, 9, 2, 5, 3, 7, 101, 18]
Index:   0  1  2  3  4  5   6   7

Step by step:
i=0: dp[0]=1, parent[0]=-1
i=1: arr[0]=10 > arr[1]=9, no update. dp[1]=1, parent[1]=-1
i=2: arr[0]=10 > arr[2]=2, arr[1]=9 > arr[2]=2, no update. dp[2]=1, parent[2]=-1
i=3: arr[2]=2 < arr[3]=5, dp[2]+1=2 > dp[3]=1. dp[3]=2, parent[3]=2
i=4: arr[2]=2 < arr[4]=3, dp[2]+1=2 > dp[4]=1. dp[4]=2, parent[4]=2
i=5: arr[2]=2 < arr[5]=7, dp[2]+1=2 > dp[5]=1. dp[5]=2, parent[5]=2
     arr[3]=5 < arr[5]=7, dp[3]+1=3 > dp[5]=2. dp[5]=3, parent[5]=3
     arr[4]=3 < arr[5]=7, dp[4]+1=3 = dp[5]=3. No change
i=6: arr[2]=2 < arr[6]=101, dp[2]+1=2. dp[6]=2, parent[6]=2
     arr[3]=5 < arr[6]=101, dp[3]+1=3 > dp[6]=2. dp[6]=3, parent[6]=3
     arr[4]=3 < arr[6]=101, dp[4]+1=3 = dp[6]=3. No change
     arr[5]=7 < arr[6]=101, dp[5]+1=4 > dp[6]=3. dp[6]=4, parent[6]=5
i=7: Similar checks... dp[7]=4, parent[7]=5

Final: dp = [1, 1, 1, 2, 2, 3, 4, 4]
       max_length = 4 at index 6
       
Reconstruction from index 6:
6 -> parent[6]=5 -> arr[5]=7
5 -> parent[5]=3 -> arr[3]=5
3 -> parent[3]=2 -> arr[2]=2
2 -> parent[2]=-1 -> stop

LIS (reversed): [2, 5, 7, 101]
```

### Complexity Analysis
- **Time Complexity:** O(n²) - nested loops comparing all pairs
- **Space Complexity:** O(n) - for dp and parent arrays
- **Best for:** Small to medium arrays (n < 5000)

---

## Algorithm 2: Binary Search - O(n log n)

### Approach
Maintain a `tails` array where `tails[i]` stores the smallest tail element for all increasing subsequences of length `i+1`. Use binary search to find where each new element fits.

### Key Insight
The `tails` array remains **sorted** at all times. For any element, we can use binary search to find where it should go, which dramatically reduces time complexity.

### How It Works

**Step 1: Initialize**
- Create empty `tails` array
- Create `parent` array to track the previous element

**Step 2: Process Each Element**
```
For each element arr[i]:
    1. Use binary search to find the position where arr[i] should go in tails
    2. If arr[i] is larger than all elements in tails:
       - Append arr[i] to tails
       - parent[i] = index of last element in tails (before appending)
    3. If arr[i] should replace an element at position pos:
       - Replace tails[pos] with arr[i] (if arr[i] is smaller)
       - Update parent tracking
```

**Step 3: Reconstruct**
- Use parent array to backtrack and build the actual LIS

### Example Trace
```
Array:  [10, 9, 2, 5, 3, 7, 101, 18]
Index:   0  1  2  3  4  5   6   7

Processing:
i=0, arr[0]=10:
   tails = [], binary_search(10) = 0 (insert position)
   tails = [10], parent[0]=-1

i=1, arr[1]=9:
   tails = [10], binary_search(9) = 0 (replace position)
   tails[0] = 9 (9 is smaller than 10, better tail)
   tails = [9], parent[1]=-1

i=2, arr[2]=2:
   tails = [9], binary_search(2) = 0 (replace position)
   tails[0] = 2
   tails = [2], parent[2]=-1

i=3, arr[3]=5:
   tails = [2], binary_search(5) = 1 (append position)
   tails = [2, 5], parent[3]=0 (arr[2]=2 is previous)

i=4, arr[4]=3:
   tails = [2, 5], binary_search(3) = 1 (replace position)
   tails[1] = 3 (3 < 5, better tail for LIS of length 2)
   tails = [2, 3], parent[4]=0

i=5, arr[5]=7:
   tails = [2, 3], binary_search(7) = 2 (append position)
   tails = [2, 3, 7], parent[5]=1 (arr[3]=5 or arr[4]=3, depending on tracking)

i=6, arr[6]=101:
   tails = [2, 3, 7], binary_search(101) = 3 (append position)
   tails = [2, 3, 7, 101], parent[6]=2

i=7, arr[7]=18:
   tails = [2, 3, 7, 101], binary_search(18) = 3 (replace position)
   tails[3] = 18 (18 < 101, better tail)
   tails = [2, 3, 7, 18], parent[7]=2

Final: Length = 4 (len(tails))

Reconstruction gives either [2, 3, 7, 101] or [2, 3, 7, 18] depending on parent tracking
```

### Complexity Analysis
- **Time Complexity:** O(n log n) - binary search for each element
- **Space Complexity:** O(n) - for tails and parent arrays
- **Best for:** Large arrays (n > 5000), needs fast computation

---

## Comparison: O(n²) vs O(n log n)

| Aspect | O(n²) DP | O(n log n) Binary Search |
|--------|----------|-------------------------|
| **Time** | Slow for large n | Fast for large n |
| **Space** | O(n) | O(n) |
| **Simplicity** | Easier to understand | More complex |
| **Best Use** | Educational, small arrays | Production, large arrays |
| **Array Size** | n ≤ 5,000 | n > 5,000 |

### Performance Example
- For n = 10,000: O(n²) ≈ 100M operations, O(n log n) ≈ 130K operations

---

## Special Cases

### 1. Empty Array
- LIS length = 0
- LIS = []

### 2. Single Element
- LIS length = 1
- LIS = [that element]

### 3. Already Sorted (Ascending)
- LIS = entire array
- LIS length = n

### 4. Reverse Sorted (Descending)
- LIS = any single element
- LIS length = 1

### 5. Array with Duplicates
- Note: "Increasing" means strictly increasing (arr[i] < arr[j], not ≤)
- Duplicates cannot be in the same LIS

### 6. Negative Numbers
- Both algorithms work with negative numbers
- Comparison logic remains the same

---

## Implementation Notes

### Key Points
1. **Strictly Increasing:** We use `<` not `≤`, so duplicates are excluded
2. **Subsequence vs Subarray:** Elements don't need to be contiguous
3. **Multiple Valid Answers:** There can be multiple LIS with the same length
4. **Parent Tracking:** Essential for reconstructing the actual sequence

### Common Pitfalls
- Confusing subsequence with subarray
- Using ≤ instead of < (which would allow duplicates)
- Not tracking parent pointers for reconstruction
- Incorrect backtracking logic

---

## Practical Applications

1. **Stock Trading:** Find the longest period of increasing prices
2. **Drug Discovery:** Match molecular sequences
3. **Database Optimization:** Query execution planning
4. **Text Processing:** Longest common subsequence variants
5. **Machine Learning:** Feature selection and sequence analysis

---

## How to Use This Solver

### Input Format
- Comma-separated: `10, 9, 2, 5, 3, 7, 101, 18`
- Space-separated: `10 9 2 5 3 7 101 18`
- Bracket notation: `[10, 9, 2, 5, 3, 7, 101, 18]`
- Negative numbers: `-5, -2, 3, 1, 4`

### Features
- **Quick Start Examples:** Pre-loaded test cases
- **Test Cases:** Predefined scenarios with expected results
- **Random Generation:** Generate random arrays for testing
- **Algorithm Comparison:** See both algorithms execute side-by-side
- **Step-by-step Trace:** Watch the algorithm work step-by-step
- **Performance Benchmarking:** Compare execution times
- **Beautiful Visualizations:** Interactive charts showing the LIS

---

## References

- **Classic DP Approach:** Cormen, Leiserson, Rivest, Stein - "Introduction to Algorithms"
- **Binary Search Optimization:** Using patience sorting concepts
- **Patent:** Related to the Robinson-Schensted correspondence in combinatorics

---

**Created:** December 2025  
**Purpose:** Educational project demonstrating fundamental algorithms  
**Algorithms:** Dynamic Programming and Binary Search optimization
