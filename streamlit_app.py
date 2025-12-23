"""
Streamlit UI for Longest Increasing Subsequence (LIS) Project

A beautiful, professional interface featuring:
- Elegant and responsive UI design
- Interactive array input with validation
- Real-time LIS computation (O(n²) and O(n log n))
- Beautiful performance comparisons
- Advanced visualizations and analytics
- Step-by-step algorithm traces

Author: DAA Semester Project
Date: December 2025
"""

import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
import io
import random
import json
import statistics
from datetime import datetime
from contextlib import redirect_stdout
import bisect
from typing import Any, Dict, List, Tuple
from lis_algorithms import lis_n2, lis_nlogn
from user_input import InputValidator, PredefinedInputs

# Session-state helpers
if "array_input" not in st.session_state:
    st.session_state.array_input = ""

if "applied_arr" not in st.session_state:
    # The array the user has explicitly confirmed via the Enter button.
    st.session_state.applied_arr = None

if "applied_input_str" not in st.session_state:
    st.session_state.applied_input_str = ""

if "input_error" not in st.session_state:
    st.session_state.input_error = ""

if "last_run" not in st.session_state:
    # Persist results across reruns so the user can scroll steps without losing output.
    st.session_state.last_run = {
        "arr": None,
        "algo": None,
        "length": None,
        "lis": None,
        "time_ms": None,
    }


def load_example(value: str) -> None:
    """Populate the input box with an example array."""
    st.session_state.array_input = value
    # Auto-apply examples so the UI updates immediately.
    arr, err = parse_array_input(value)
    st.session_state.input_error = err
    st.session_state.applied_arr = arr if not err else None
    st.session_state.applied_input_str = value


def load_random_example(size: int, seed: int | None = None) -> None:
    """Generate a random array and load it into the input."""
    if seed is not None:
        random.seed(seed)

    # Keep the range simple and consistent with the rest of the project.
    arr = [random.randint(1, size * 2) for _ in range(size)]
    st.session_state.array_input = " ".join(map(str, arr))
    st.session_state.input_error = ""
    st.session_state.applied_arr = arr
    st.session_state.applied_input_str = st.session_state.array_input


def _reconstruct_lis_from_parent(arr: List[int], parent: List[int], end_index: int) -> List[int]:
    lis = []
    current = end_index
    while current != -1:
        lis.append(arr[current])
        current = parent[current]
    lis.reverse()
    return lis


def trace_lis_n2(arr: List[int]) -> Tuple[int, List[int], List[Dict[str, Any]]]:
    """Trace O(n²) DP LIS step-by-step for explanation in Streamlit."""
    if not arr:
        return 0, [], [{"title": "Empty input", "description": "Array is empty, LIS length is 0.", "dp": [], "parent": []}]

    n = len(arr)
    dp = [1] * n
    parent = [-1] * n
    steps: List[Dict[str, Any]] = [
        {
            "title": "Initialize",
            "description": "Start with dp[i]=1 for all i (each element alone is an LIS of length 1).",
            "dp": dp.copy(),
            "parent": parent.copy(),
        }
    ]

    for i in range(1, n):
        steps.append(
            {
                "title": f"i = {i} (value={arr[i]})",
                "description": "Compare arr[i] with all previous elements arr[j] where j < i.",
                "i": i,
                "dp": dp.copy(),
                "parent": parent.copy(),
            }
        )
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
                steps.append(
                    {
                        "title": f"Update at i={i}, j={j}",
                        "description": (
                            f"Because {arr[j]} < {arr[i]}, we can extend the subsequence ending at j. "
                            f"Update dp[{i}] = dp[{j}] + 1 = {dp[i]} and set parent[{i}] = {j}."
                        ),
                        "i": i,
                        "j": j,
                        "dp": dp.copy(),
                        "parent": parent.copy(),
                    }
                )

    max_length = max(dp)
    max_index = dp.index(max_length)
    steps.append(
        {
            "title": "Find best LIS end",
            "description": f"The maximum dp value is {max_length} at index {max_index}. Now backtrack using parent[].",
            "dp": dp.copy(),
            "parent": parent.copy(),
            "end_index": max_index,
        }
    )

    lis_indices = []
    cur = max_index
    while cur != -1:
        lis_indices.append(cur)
        steps.append(
            {
                "title": "Backtrack",
                "description": f"Pick arr[{cur}] = {arr[cur]} and move to parent[{cur}] = {parent[cur]}",
                "dp": dp.copy(),
                "parent": parent.copy(),
                "picked_index": cur,
            }
        )
        cur = parent[cur]

    lis_indices.reverse()
    lis = [arr[k] for k in lis_indices]
    steps.append(
        {
            "title": "Final LIS",
            "description": f"LIS indices = {lis_indices} → LIS = {lis}",
            "dp": dp.copy(),
            "parent": parent.copy(),
            "lis": lis,
        }
    )

    return max_length, lis, steps


def trace_lis_nlogn(arr: List[int]) -> Tuple[int, List[int], List[Dict[str, Any]]]:
    """Trace O(n log n) LIS (tails + binary search) step-by-step for explanation in Streamlit."""
    if not arr:
        return 0, [], [{"title": "Empty input", "description": "Array is empty, LIS length is 0.", "tails": []}]

    n = len(arr)
    tails: List[int] = []
    parent = [-1] * n
    lis_index: List[int] = []

    steps: List[Dict[str, Any]] = [
        {
            "title": "Initialize",
            "description": "Start with empty tails[]. tails[k] stores the smallest tail value of an increasing subsequence of length k+1.",
            "tails": tails.copy(),
            "parent": parent.copy(),
            "lis_index": lis_index.copy(),
        }
    ]

    for i, x in enumerate(arr):
        pos = bisect.bisect_left(tails, x)
        if pos > 0:
            parent[i] = lis_index[pos - 1]

        action: str
        if pos == len(tails):
            tails.append(x)
            lis_index.append(i)
            action = "extend"
        else:
            tails[pos] = x
            lis_index[pos] = i
            action = "replace"

        steps.append(
            {
                "title": f"Process i={i} (value={x})",
                "description": (
                    f"Binary search position = {pos}. Action: {action}. "
                    "If replace, we keep the same length but make the tail smaller (better for future extensions)."
                ),
                "i": i,
                "value": x,
                "pos": pos,
                "action": action,
                "tails": tails.copy(),
                "parent": parent.copy(),
                "lis_index": lis_index.copy(),
            }
        )

    length = len(tails)
    end_index = lis_index[-1] if lis_index else -1
    lis = _reconstruct_lis_from_parent(arr, parent, end_index)
    steps.append(
        {
            "title": "Reconstruct",
            "description": f"LIS length is len(tails) = {length}. Backtrack from index {end_index} using parent[] to build the LIS.",
            "tails": tails.copy(),
            "parent": parent.copy(),
            "lis_index": lis_index.copy(),
            "lis": lis,
        }
    )

    return length, lis, steps


# Page configuration
st.set_page_config(
    page_title="LIS Algorithm Solver | Professional Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Longest Increasing Subsequence Solver - Professional UI Edition"}
)

# Professional Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #2ca02c;
        --accent-color: #ff7f0e;
        --danger-color: #d62728;
        --background-light: #f8f9fa;
        --border-color: #e0e0e0;
    }

    /* Main container styling */
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Header styling */
    h1 {
        color: #1f77b4;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #34495e;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }

    /* Card containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 2px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.03) 0%, rgba(44, 160, 44, 0.03) 100%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-color);
    }

    /* Buttons - Premium styling */
    div.stButton > button {
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #1f77b4 0%, #1a5fa0 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }

    div.stButton > button:hover {
        box-shadow: 0 4px 16px rgba(31, 119, 180, 0.5);
        transform: translateY(-2px);
    }

    /* Input fields - Modern styling */
    input, textarea, select {
        border-radius: 10px !important;
        border: 2px solid var(--border-color) !important;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #0052cc !important;
        box-shadow: 0 0 0 3px rgba(0, 82, 204, 0.1);
    }

    /* Metrics - Enhanced styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #0052cc;
        box-shadow: 0 2px 8px rgba(0, 82, 204, 0.08);
    }

    /* Tabs - Modern styling */
    div[data-testid="stTabs"] button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    /* Success/Error messages */
    .element-container .stAlert {
        border-radius: 12px;
        border: 2px solid;
    }

    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary-color), transparent);
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 2px solid var(--border-color);
    }

    /* Code blocks - Beautiful styling */
    code {
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f3f7 100%);
        border-radius: 8px;
        padding: 0.2rem 0.6rem;
        font-weight: 500;
    }

    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Sidebar button styling */
    div[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }

    /* Better spacing for sections */
    .section-spacing {
        margin-top: 2.5rem;
        margin-bottom: 2rem;
    }

    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* Loading spinner */
    .stSpinner {
        text-align: center;
    }

    /* Icons styling */
    .icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }

    /* Professional caption */
    .caption-pro {
        color: #95a5a6;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def parse_array_input(input_str: str) -> Tuple[List[int], str]:
    """
    Parse user input string to array.
    
    Args:
        input_str: Comma or space separated integers
        
    Returns:
        Tuple of (parsed_array, error_message)
    """
    if not input_str or not input_str.strip():
        return [], ""

    # Reuse the project's validator (supports brackets, mixed commas/spaces, etc.).
    is_valid, parsed, err = InputValidator.validate_array_string(input_str)
    if not is_valid or parsed is None:
        msg = err.strip() if isinstance(err, str) else "Invalid input"
        if not msg:
            msg = "Invalid input. Use integers separated by spaces/commas (brackets allowed)."
        return [], msg

    if len(parsed) > 10000:
        return [], "Array size too large (max 10000 elements)"

    return parsed, ""


def _is_strictly_increasing(seq: List[int]) -> bool:
    return all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))


def _is_subsequence(arr: List[int], subseq: List[int]) -> bool:
    if not subseq:
        return True
    j = 0
    for x in arr:
        if x == subseq[j]:
            j += 1
            if j == len(subseq):
                return True
    return False


def _benchmark(func, arr: List[int], runs: int, warmup: bool = True) -> List[float]:
    if warmup:
        with redirect_stdout(io.StringIO()):
            func(arr)
    times: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            func(arr)
        times.append((time.perf_counter() - start) * 1000)
    return times


def _apply_input_string(value: str) -> None:
    """Apply an input string into the app's 'applied' state (and update the input widget)."""
    parsed, err = parse_array_input(value)
    st.session_state.input_error = err
    st.session_state.applied_arr = parsed if (not err and len(parsed) > 0) else None
    st.session_state.applied_input_str = value


def _apply_current_input() -> None:
    """Apply whatever is currently typed in the input widget."""
    value = st.session_state.get("array_input", "")
    parsed, err = parse_array_input(value)
    st.session_state.input_error = err
    st.session_state.applied_arr = parsed if (not err and len(parsed) > 0) else None
    st.session_state.applied_input_str = value


def _generate_and_apply_random(size: int, min_value: int, max_value: int) -> None:
    gen_arr = [random.randint(int(min_value), int(max_value)) for _ in range(int(size))]
    value = " ".join(map(str, gen_arr))
    # Safe here because this runs as a button callback.
    st.session_state.array_input = value
    _apply_input_string(value)


def _generate_sidebar_random() -> None:
    """Generate random array from sidebar controls and apply it."""
    size = int(st.session_state.get("rand_size", 20))
    min_value = int(st.session_state.get("rand_min", 1))
    max_value = int(st.session_state.get("rand_max", 40))

    if max_value < min_value:
        st.session_state.input_error = "Random range: Max must be ≥ Min."
        st.session_state.applied_arr = None
        return

    _generate_and_apply_random(size, min_value, max_value)


def plot_lis_visualization(arr: List[int], lis_result: List[int]) -> go.Figure:
    """Create beautiful interactive visualization of LIS."""
    fig = go.Figure()
    
    # Find LIS indices - CORRECTLY match LIS elements to their positions in original array
    # Important: We need to find the sequence of INDICES that produces the LIS values
    # in the correct order, maintaining strictly increasing indices
    
    if not lis_result:
        # If no LIS, all elements are gray
        lis_indices = []
    else:
        # Find the indices in arr that form the LIS
        # We need to find positions where each LIS value appears
        # and ensure indices are strictly increasing
        lis_indices = []
        last_idx = -1
        
        for lis_val in lis_result:
            # Find the first occurrence of lis_val AFTER last_idx
            for i in range(last_idx + 1, len(arr)):
                if arr[i] == lis_val:
                    lis_indices.append(i)
                    last_idx = i
                    break
    
    # Create colors: non-LIS (light gray) vs LIS (green)
    colors = ["#2ca02c" if i in lis_indices else "#e8e8e8" for i in range(len(arr))]
    
    # Add bars for original array
    fig.add_trace(go.Bar(
        x=list(range(len(arr))),
        y=arr,
        name="Array Elements",
        marker=dict(color=colors, line=dict(color="#1f77b4", width=2)),
        text=arr,
        textposition="outside",
        hovertemplate="<b>Index: %{x}</b><br>Value: %{y}<extra></extra>",
    ))
    
    # Add line connecting LIS elements
    if lis_indices:
        fig.add_trace(go.Scatter(
            x=lis_indices,
            y=[arr[i] for i in lis_indices],
            mode='lines+markers',
            name="LIS Path",
            line=dict(color='#d62728', width=3, dash='solid'),
            marker=dict(size=12, color='#d62728', symbol='circle', line=dict(color='#ffffff', width=2)),
            hovertemplate="<b>LIS Element</b><br>Index: %{x}<br>Value: %{y}<extra></extra>",
        ))
    
    fig.update_layout(
        title={
            'text': f"📊 LIS Visualization | Array Length: {len(arr)} | LIS Length: {len(lis_result)}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#1f77b4'}
        },
        xaxis_title="📍 Index",
        yaxis_title="📈 Value",
        hovermode="x unified",
        height=450,
        barmode="overlay",
        plot_bgcolor='rgba(245, 247, 250, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12, family="Arial"),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='#e0e0e0', borderwidth=1)
    )
    
    return fig


def plot_performance_comparison(sizes: List[int], n2_times: List[float], nlogn_times: List[float]) -> go.Figure:
    """Create beautiful performance comparison chart."""
    fig = go.Figure()
    
    # Add O(n²) trace
    fig.add_trace(go.Scatter(
        x=sizes, y=n2_times,
        mode='lines+markers',
        name='O(n²) DP Algorithm',
        line=dict(color='#d62728', width=4),
        marker=dict(size=12, symbol='circle', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)',
        hovertemplate="<b>O(n²)</b><br>Size: %{x}<br>Time: %{y:.4f} ms<extra></extra>",
    ))
    
    # Add O(n log n) trace
    fig.add_trace(go.Scatter(
        x=sizes, y=nlogn_times,
        mode='lines+markers',
        name='O(n log n) Binary Search',
        line=dict(color='#2ca02c', width=4),
        marker=dict(size=12, symbol='square', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.2)',
        hovertemplate="<b>O(n log n)</b><br>Size: %{x}<br>Time: %{y:.4f} ms<extra></extra>",
    ))
    
    fig.update_layout(
        title={
            'text': "⏱️ Algorithm Performance Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#1f77b4'}
        },
        xaxis_title="📏 Array Size (n)",
        yaxis_title="⏱️ Time (milliseconds)",
        hovermode="x unified",
        height=450,
        plot_bgcolor='rgba(245, 247, 250, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12, family="Arial"),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='#e0e0e0', borderwidth=1),
        yaxis_type="log"
    )
    
    return fig


def main():
    """Main Streamlit application with professional UI."""

    # Custom CSS for beautiful styling with theme-aware colors
    st.markdown("""
    <style>
    /* Main app background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e8f4f8 100%);
        min-height: 100vh;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f172a 0%, #1a2e4a 100%);
        }
    }
    
    /* Section containers with subtle backgrounds */
    [data-testid="stContainer"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stContainer"] {
            background-color: rgba(45, 45, 60, 0.9);
        }
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 12px;
    }
    
    /* Section headers with gradient background */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding: 0.8rem 1rem;
        background: linear-gradient(90deg, #0052cc 0%, #00a3e0 100%);
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 82, 204, 0.3);
    }
    
    @media (prefers-color-scheme: dark) {
        .section-header {
            background: linear-gradient(90deg, #0052cc 0%, #00b8e6 100%);
            box-shadow: 0 4px 15px rgba(0, 82, 204, 0.4);
        }
    }
    
    /* Example card styling */
    .example-cards {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    
    /* Custom button styling */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        margin: 1.5rem 0;
    }
    
    /* Input field styling */
    input[type="text"], input[type="number"], select {
        background-color: #f8f9fa !important;
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    input[type="text"]:focus, input[type="number"]:focus, select:focus {
        border-color: #0052cc !important;
        box-shadow: 0 0 0 3px rgba(0, 82, 204, 0.1) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        input[type="text"], input[type="number"], select {
            background-color: #1a2e4a !important;
            border-color: #00b8e6 !important;
            color: white !important;
        }
    }
    
    /* Container with border styling */
    [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 249, 255, 0.95) 100%);
        border-left: 4px solid #0052cc;
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
            background: linear-gradient(135deg, rgba(26, 46, 74, 0.95) 0%, rgba(20, 40, 70, 0.95) 100%);
            border-left-color: #00b8e6;
        }
    }
    
    /* Beautiful Navbar Styling */
    .navbar-container {
        background: linear-gradient(90deg, #0052cc 0%, #00a3e0 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 82, 204, 0.25),
                    0 2px 8px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .navbar-content {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .navbar-title {
        color: white;
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.3px;
    }
    
    .navbar-subtitle {
        color: rgba(255, 255, 255, 0.85);
        margin: 0;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: right;
        line-height: 1.3;
    }
    
    .navbar-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-right: 1rem;
        border-right: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .navbar-badge {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .navbar-badge.premium {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.3), rgba(255, 152, 0, 0.3));
        border-color: rgba(255, 193, 7, 0.5);
    }
    
    /* Project Title Badge */
    .project-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-right: 0.8rem;
        backdrop-filter: blur(5px);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .project-badge .highlight {
        color: #fbbf24;
        font-weight: 700;
    }
    
    /* Tagline styling */
    .tagline-container {
        text-align: left;
        margin: 0.3rem auto 1rem auto;
        padding: 0.8rem 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 0.8rem;
    }
    
    .tagline-main {
        font-size: 35px;
        font-weight: 700;
        background: linear-gradient(90deg, #0052cc 0%, #00a3e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: 0.5px;
        display: flex;
        gap: 1.5rem;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .tagline-sub {
        color: #4b5563;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.8;
        max-width: 900px;
        width: 100%;
        padding: 0;
    }
    
    @media (prefers-color-scheme: dark) {
        .navbar-container {
            background: linear-gradient(90deg, #0052cc 0%, #00a3e0 100%);
            box-shadow: 0 8px 32px rgba(0, 82, 204, 0.25),
                        0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .tagline-main {
            background: linear-gradient(90deg, #00b8e6 0%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .tagline-sub {
            color: #cbd5e1;
        }
    }
    </style>
    """, unsafe_allow_html=True)


    # Beautiful Navbar Header
    st.markdown("""
    <div class="navbar-container">
        <div class="navbar-content">
            <div class="navbar-brand">
                <span style="font-size: 2.5rem;">📊</span>
                <h1 class="navbar-title">Longest Increasing Subsequence</h1>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tagline Section
    st.markdown("""
    <div class="tagline-container">
        <p class="tagline-main">⚡ Fast • 📈 Beautiful • 🎓 Educational</p>
        <p class="tagline-sub">
            Compare <strong>O(n²)</strong> and <strong>O(n log n)</strong> LIS algorithms with interactive visualizations,<br/>
            performance benchmarking, and step-by-step algorithm traces.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Test Cases - Simplified
    st.divider()
    st.markdown("<div class='section-header'>📚 Test Cases</div>", unsafe_allow_html=True)
    
    sample_ids = list(PredefinedInputs.SAMPLES.keys())
    sample_labels = [f"{PredefinedInputs.SAMPLES[k]['name']}" for k in sample_ids]
    selected_label = st.selectbox("Choose a preset:", options=sample_labels, label_visibility="collapsed")
    
    for sid in sample_ids:
        if PredefinedInputs.SAMPLES[sid]['name'] == selected_label:
            sample = PredefinedInputs.get_sample_info(sid)
            break
    
    if sample:
        st.markdown(f"**{sample.get('description', '')}**")
        st.code(sample["array"])
        if st.button("✅ Load", key="load_sample_case", use_container_width=True):
            load_example(" ".join(map(str, sample["array"])))

    # Main content
    st.divider()

    # Input section
    with st.container(border=True):
        st.markdown("## 📥 Input Array")
        
        # Simpler input with better visual hierarchy
        user_input = st.text_area(
            "Enter integers (comma or space separated)",
            placeholder="Example: 10 9 2 5 3 7 101 18",
            help="Separate numbers with spaces or commas",
            key="array_input",
            height=80,
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("✓ Enter Array", key="enter_input", use_container_width=True, on_click=_apply_current_input)
        with col_btn2:
            with st.expander("🎲 Generate Random", expanded=False):
                rg1, rg2, rg3 = st.columns(3)
                with rg1:
                    g_size = st.number_input("Size", min_value=1, max_value=10000, value=10, step=1, key="rg_size")
                with rg2:
                    g_min = st.number_input("Min", value=1, step=1, key="rg_min")
                with rg3:
                    g_max = st.number_input("Max", value=30, step=1, key="rg_max")

                if g_max < g_min:
                    st.error("❌ Max must be ≥ Min")
                else:
                    st.button(
                        "🔄 Generate",
                        key="rg_generate",
                        use_container_width=True,
                        on_click=_generate_and_apply_random,
                        args=(int(g_size), int(g_min), int(g_max)),
                    )

        if st.session_state.input_error:
            st.error(f"❌ {st.session_state.input_error}")
            st.stop()

        if st.session_state.applied_arr is None:
            st.info("💡 Tip: Enter an array or generate a random one to start")
            st.stop()

        if st.session_state.applied_input_str != st.session_state.array_input:
            st.warning("⚠️ Array changed. Click **Enter Array** to apply.")

    arr = st.session_state.applied_arr

    # Input summary
    with st.container(border=True):
        st.markdown("## 📊 Array Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📏 Size", len(arr))
        c2.metric("📉 Min Value", min(arr))
        c3.metric("📈 Max Value", max(arr))
        c4.metric("📊 Range", max(arr) - min(arr))
        st.code(arr, language="python")

    # Algorithm selection and execution
    with st.container(border=True):
        st.markdown("## ⚙️ Run Algorithm")
        col1, col2 = st.columns([3, 1])
        with col1:
            algo_choice = st.radio(
                "Choose Algorithm:",
                ["O(n²) - Simple DP", "O(n log n) - Binary Search", "Compare Both"],
                horizontal=True,
            )
        with col2:
            st.write("")
            run_button = st.button("▶️ Run", key="run_one", use_container_width=True)

    if run_button:
        # Execute algorithm(s)
        if "Compare Both" in algo_choice:
            # Run both and store both
            with st.spinner("🔄 Running O(n²) DP Algorithm..."):
                start_time = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    length_n2, lis_result_n2 = lis_n2(arr)
                elapsed_ms_n2 = (time.perf_counter() - start_time) * 1000

            with st.spinner("🔄 Running O(n log n) Binary Search Algorithm..."):
                start_time = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    length_nlogn, lis_result_nlogn = lis_nlogn(arr)
                elapsed_ms_nlogn = (time.perf_counter() - start_time) * 1000

            st.session_state.last_run = {
                "arr": arr.copy(),
                "algo": algo_choice,
                "length_n2": length_n2,
                "lis_n2": lis_result_n2,
                "time_ms_n2": elapsed_ms_n2,
                "length_nlogn": length_nlogn,
                "lis_nlogn": lis_result_nlogn,
                "time_ms_nlogn": elapsed_ms_nlogn,
            }
        elif algo_choice.startswith("O(n²)"):
            with st.spinner("🔄 Running O(n²) DP Algorithm..."):
                start_time = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    length, lis_result = lis_n2(arr)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

            st.session_state.last_run = {
                "arr": arr.copy(),
                "algo": algo_choice,
                "length": length,
                "lis": lis_result,
                "time_ms": elapsed_ms,
            }
        else:
            with st.spinner("🔄 Running O(n log n) Binary Search Algorithm..."):
                start_time = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    length, lis_result = lis_nlogn(arr)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

            st.session_state.last_run = {
                "arr": arr.copy(),
                "algo": algo_choice,
                "length": length,
                "lis": lis_result,
                "time_ms": elapsed_ms,
            }
        
        st.success("✅ Algorithm executed successfully!")

    last = st.session_state.last_run
    has_result = isinstance(last.get("arr"), list) and last.get("arr") == arr

    if not has_result:
        st.info("💡 Select an algorithm and click **Run** to see results.")
    else:
        # Results section
        if "Compare Both" in last.get("algo", ""):
            with st.container(border=True):
                st.markdown("## 📊 Comparison Results")
                
                tab1, tab2, tab3 = st.tabs(["O(n²) Results", "O(n log n) Results", "Head-to-Head"])
                
                with tab1:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("✓ LIS Length", int(last["length_n2"]))
                    c2.metric("⏱️ Time (ms)", f"{last['time_ms_n2']:.4f}")
                    c3.metric("📦 Algorithm", "O(n²)")
                    st.markdown("**LIS Sequence:**")
                    st.code(last["lis_n2"], language="python")
                
                with tab2:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("✓ LIS Length", int(last["length_nlogn"]))
                    c2.metric("⏱️ Time (ms)", f"{last['time_ms_nlogn']:.4f}")
                    c3.metric("📦 Algorithm", "O(n log n)")
                    st.markdown("**LIS Sequence:**")
                    st.code(last["lis_nlogn"], language="python")
                
                with tab3:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔴 O(n²) Time", f"{last['time_ms_n2']:.4f} ms")
                    c2.metric("🟢 O(n log n) Time", f"{last['time_ms_nlogn']:.4f} ms")
                    c3.metric("✓ Same Length?", "Yes" if last['length_n2'] == last['length_nlogn'] else "❌ No")
                    
                    # Comparison chart
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Bar(
                        x=["O(n²)", "O(n log n)"],
                        y=[last['time_ms_n2'], last['time_ms_nlogn']],
                        marker=dict(color=["#d62728", "#2ca02c"]),
                        text=[f"{last['time_ms_n2']:.4f} ms", f"{last['time_ms_nlogn']:.4f} ms"],
                        textposition="outside"
                    ))
                    fig_comp.update_layout(
                        title="⏱️ Execution Time Comparison",
                        yaxis_title="Time (milliseconds)",
                        height=400,
                        showlegend=False,
                        hovermode="x"
                    )
                    st.plotly_chart(fig_comp, use_container_width=True, key="comp_chart")
        else:
            with st.container(border=True):
                st.markdown("## 📊 Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("✓ LIS Length", int(last["length"]))
                c2.metric("⏱️ Time (ms)", f"{last['time_ms']:.4f}")
                algo_name = "O(n²)" if "O(n²)" in last["algo"] else "O(n log n)"
                c3.metric("📦 Algorithm", algo_name)
                st.markdown("**LIS Sequence:**")
                st.code(last["lis"], language="python")

        # Download section
        with st.container(border=True):
            st.markdown("## 💾 Download Results")
            if "Compare Both" in last.get("algo", ""):
                payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "input": arr,
                    "results": {
                        "n2_algorithm": {
                            "length": int(last["length_n2"]),
                            "lis": last["lis_n2"],
                            "time_ms": float(last["time_ms_n2"]),
                        },
                        "nlogn_algorithm": {
                            "length": int(last["length_nlogn"]),
                            "lis": last["lis_nlogn"],
                            "time_ms": float(last["time_ms_nlogn"]),
                        }
                    }
                }
            else:
                payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "input": arr,
                    "algorithm": last["algo"],
                    "lis_length": int(last["length"]),
                    "lis": last["lis"],
                    "time_ms": float(last["time_ms"]),
                }
            
            json_bytes = json.dumps(payload, indent=2).encode("utf-8")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📥 JSON Results",
                    data=json_bytes,
                    file_name="lis_result.json",
                    mime="application/json",
                    width="stretch",
                )
            with col2:
                st.download_button(
                    "📥 LIS Sequence",
                    data=" ".join(map(str, last.get("lis", last.get("lis_n2", [])))).encode("utf-8"),
                    file_name="lis_sequence.txt",
                    mime="text/plain",
                    width="stretch",
                )
            with col3:
                st.download_button(
                    "📥 Input Array",
                    data=" ".join(map(str, arr)).encode("utf-8"),
                    file_name="input_array.txt",
                    mime="text/plain",
                    width="stretch",
                )

        # Visualization section
        with st.container(border=True):
            st.markdown("## 📈 Visualization")
            if "Compare Both" in last.get("algo", ""):
                # Show both visualizations
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = plot_lis_visualization(arr, last["lis_n2"])
                    st.plotly_chart(fig1, use_container_width=True, key="viz_n2")
                with col2:
                    fig2 = plot_lis_visualization(arr, last["lis_nlogn"])
                    st.plotly_chart(fig2, use_container_width=True, key="viz_nlogn")
            else:
                fig = plot_lis_visualization(arr, last.get("lis", last.get("lis_n2", [])))
                st.plotly_chart(fig, use_container_width=True, key="viz_single")

        # Performance analysis
        with st.container(border=True):
            st.markdown("## 🔍 Benchmark & Correctness")
            st.markdown("Advanced tools for performance analysis and algorithm verification.")

            col1, col2, col3 = st.columns(3)
            with col1:
                bench_runs = st.number_input("Runs", min_value=1, max_value=50, value=5, step=1, key="bench_runs")
            with col2:
                warmup = st.checkbox("Warm-up", value=True, key="bench_warmup")
            with col3:
                compare_both_bench = st.checkbox("Compare Both", value=False, key="bench_compare")

            do_check = st.checkbox("✓ Correctness Check", value=True, key="correctness_check")
            run_bench = st.button("🚀 Run Benchmark", key="run_benchmark", width="stretch")

            if run_bench:
                if len(arr) > 5000 and (algo_choice.startswith("O(n²)") or compare_both_bench):
                    st.warning("⚠️ O(n²) on large arrays can be slow. Consider a smaller array.")

                if compare_both_bench or "Compare Both" in last.get("algo", ""):
                    times_n2 = _benchmark(lis_n2, arr, int(bench_runs), warmup=bool(warmup))
                    times_nlogn = _benchmark(lis_nlogn, arr, int(bench_runs), warmup=bool(warmup))
                    
                    avg_n2 = statistics.mean(times_n2)
                    avg_nlogn = statistics.mean(times_nlogn)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🔴 O(n²) Avg", f"{avg_n2:.4f} ms")
                    m2.metric("🟢 O(n log n) Avg", f"{avg_nlogn:.4f} ms")
                    m3.metric("📊 Runs", int(bench_runs))
                    
                    st.markdown(f"**Raw times (ms):** {', '.join(f'{t:.4f}' for t in times_n2[:5])}..." if len(times_n2) > 5 else f"**Raw times (ms):** {', '.join(f'{t:.4f}' for t in times_n2)}")
                else:
                    if algo_choice.startswith("O(n²)"):
                        times = _benchmark(lis_n2, arr, int(bench_runs), warmup=bool(warmup))
                    else:
                        times = _benchmark(lis_nlogn, arr, int(bench_runs), warmup=bool(warmup))

                    avg_ms = statistics.mean(times)
                    min_ms = min(times)
                    max_ms = max(times)
                    std_ms = statistics.pstdev(times) if len(times) > 1 else 0.0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Avg (ms)", f"{avg_ms:.4f}")
                    m2.metric("Min (ms)", f"{min_ms:.4f}")
                    m3.metric("Max (ms)", f"{max_ms:.4f}")
                    m4.metric("Std (ms)", f"{std_ms:.4f}")

                if do_check:
                    with st.spinner("⏳ Running correctness check..."):
                        with redirect_stdout(io.StringIO()):
                            len_n2, lis_seq_n2 = lis_n2(arr)
                            len_nlogn, lis_seq_nlogn = lis_nlogn(arr)

                    ok_subseq_n2 = _is_subsequence(arr, lis_seq_n2) and _is_strictly_increasing(lis_seq_n2)
                    ok_subseq_nlogn = _is_subsequence(arr, lis_seq_nlogn) and _is_strictly_increasing(lis_seq_nlogn)

                    st.divider()
                    if len_n2 != len_nlogn:
                        st.error(f"❌ Length mismatch: O(n²)={len_n2} vs O(n log n)={len_nlogn}")
                    else:
                        st.success(f"✅ Both algorithms agree: LIS length = {len_n2}")

                    if not ok_subseq_n2:
                        st.warning("⚠️ O(n²) result is not a valid increasing subsequence")
                    if not ok_subseq_nlogn:
                        st.warning("⚠️ O(n log n) result is not a valid increasing subsequence")

                    if ok_subseq_n2 and ok_subseq_nlogn:
                        st.success("✅ Both sequences are valid increasing subsequences!")

    # Step-by-step visualization section
    with st.container(border=True):
        st.markdown("## 👣 Step-by-Step Trace")
        st.markdown("Watch how the algorithm builds the LIS step-by-step. Best with arrays ≤ 50 elements.")

        if len(arr) > 50:
            st.warning("⚠️ Step-by-step tracing works best with ≤ 50 elements. Your array has {}.".format(len(arr)))
            return

        tab_dp, tab_bs = st.tabs(["🔴 O(n²) DP Algorithm", "🟢 O(n log n) Binary Search"])

        with tab_dp:
            dp_len, dp_lis, dp_steps = trace_lis_n2(arr)
            step_idx = st.slider("Choose step", 0, len(dp_steps) - 1, 0, key="dp_step")
            step = dp_steps[step_idx]
            
            # Better step visualization
            st.markdown(f"### **Step {step_idx + 1}:** {step['title']}")
            st.info(f"💡 {step['description']}")
            
            # Show array progress
            st.markdown("**Current State:**")
            st.write(f"🔢 **dp array** (LIS length up to each index):")
            st.code(str(step.get('dp', [])))
            
            # Show current indices if available
            if "i" in step or "j" in step:
                st.divider()
                st.markdown("**Current Comparison:**")
                metric_cols = st.columns(2)
                if "i" in step:
                    with metric_cols[0]:
                        st.metric("Index i", f"arr[{step.get('i')}] = {arr[step.get('i')] if step.get('i') < len(arr) else 'N/A'}")
                if "j" in step:
                    with metric_cols[1]:
                        st.metric("Index j", f"arr[{step.get('j')}] = {arr[step.get('j')] if step.get('j') < len(arr) else 'N/A'}")
            
            if step_idx == len(dp_steps) - 1:
                st.success(f"✅ **Complete!** LIS length = **{dp_len}**, Sequence = **{dp_lis}**")

        with tab_bs:
            bs_len, bs_lis, bs_steps = trace_lis_nlogn(arr)
            step_idx = st.slider("Choose step", 0, len(bs_steps) - 1, 0, key="bs_step")
            step = bs_steps[step_idx]
            
            # Better step visualization
            st.markdown(f"### **Step {step_idx + 1}:** {step['title']}")
            st.info(f"💡 {step['description']}")
            
            # Show arrays
            st.markdown("**Current State:**")
            st.write(f"📊 **tails array** (smallest ending value for each LIS length):")
            st.code(str(step.get('tails', [])))
            
            # Show current action
            if "action" in step:
                st.divider()
                st.markdown("**Current Action:**")
                if step.get('action').upper() == 'INSERT':
                    st.success(f"➕ Inserting new element")
                elif step.get('action').upper() == 'APPEND':
                    st.success(f"➕ Appending to extend LIS")
                else:
                    st.info(f"🔍 {step.get('action').upper()}")
            
            if "i" in step:
                st.markdown(f"**Current Element:** arr[{step.get('i')}] = **{arr[step.get('i')] if step.get('i') < len(arr) else 'N/A'}**")
            
            if step_idx == len(bs_steps) - 1:
                st.success(f"✅ **Complete!** LIS length = **{bs_len}**, Sequence = **{bs_lis}**")

    st.divider()
    st.markdown("<p style='text-align: center; color: #95a5a6; font-size: 0.9rem;'><b>© 2025 DAA Semester Project</b> | Professional Edition v2.0</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
