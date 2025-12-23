# Longest Increasing Subsequence (LIS) Solver

A professional, interactive web application for learning and comparing two LIS algorithms with beautiful visualizations.

## Features

⚡ **Fast** - Compare O(n²) and O(n log n) algorithms  
📈 **Beautiful** - Interactive visualizations with Plotly  
🎓 **Educational** - Step-by-step algorithm traces and analysis

## Quick Start

### Setup
```bash
pip install -r requirements.txt
python quick_start.py
```

The app will open at `http://localhost:8501`

### Usage
1. **Enter Array:** Input numbers (comma, space, or bracket separated)
   - Example: `10 9 2 5 3 7 101 18`
   - Supports negative numbers: `-5, -2, 3, 1, 4`

2. **Choose Algorithm:**
   - O(n²) Dynamic Programming
   - O(n log n) Binary Search
   - Compare both side-by-side

3. **View Results:**
   - LIS sequence and length
   - Execution time comparison
   - Interactive visualizations
   - Step-by-step algorithm trace

## File Structure

```
lis-simulator/
├── lis_algorithms.py          # Core algorithm implementations
├── user_input.py              # Input validation & test cases
├── streamlit_app.py           # Web UI application
├── quick_start.py             # Startup script
├── requirements.txt           # Python dependencies
├── ALGORITHM_GUIDE.md         # Complete algorithm explanation
└── README.md                  # This file
```

## Algorithms

### O(n²) Dynamic Programming
- **Best for:** Small arrays (n < 5,000)
- **Time:** O(n²), Space: O(n)
- **Approach:** Compare each element with all previous elements

### O(n log n) Binary Search
- **Best for:** Large arrays (n > 5,000)
- **Time:** O(n log n), Space: O(n)
- **Approach:** Maintain sorted array of tail values, use binary search

**For detailed explanation:** See [ALGORITHM_GUIDE.md](ALGORITHM_GUIDE.md)

## Examples

### Basic Example
```
Input:  [10, 9, 2, 5, 3, 7, 101, 18]
Output: [2, 3, 7, 101] (length: 4)
```

### Edge Cases Supported
- **Already Sorted:** [1, 2, 3, 4, 5] → [1, 2, 3, 4, 5]
- **Reverse Sorted:** [5, 4, 3, 2, 1] → [5] (or any single element)
- **With Negatives:** [-5, -2, 3, 1, 4] → [-5, -2, 3, 4]
- **Duplicates:** [1, 1, 2, 3] → [1, 2, 3] (strictly increasing)

## Technologies

- **Python 3.12** - Core language
- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical operations

## Requirements

See `requirements.txt` for all dependencies:
- streamlit
- plotly
- numpy
- matplotlib

## Performance Comparison

| n (Size) | O(n²) Time | O(n log n) Time | Speedup |
|----------|-----------|-----------------|---------|
| 100 | 0.1ms | 0.05ms | 2x |
| 1,000 | 10ms | 0.3ms | 33x |
| 10,000 | 1000ms | 4ms | 250x |

## Project

December 2025 - Educational DAA Project

---

**For complete algorithm explanation:** Read [ALGORITHM_GUIDE.md](ALGORITHM_GUIDE.md)
