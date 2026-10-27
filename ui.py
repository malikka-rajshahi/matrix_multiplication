import time

import pandas as pd
import streamlit as st

from main import parse_input_text
from performance_evaluations import METHODS
from ui_helpers import (
    EVAL_MAX_K,
    build_evaluation1,
    build_evaluation2,
    clear_evaluation_caches,
    latex_matrix,
    show_evaluation_tables,
    show_figure,
)

PAGE_METHODS = ("Direct", "Strassen")
MAX_EDITOR_N = 7
MAX_DISPLAY_N = 8

EVAL_CAP_NOTE = f"""
**Evaluations are capped at k = {EVAL_MAX_K} (n = 1 … {2 ** EVAL_MAX_K}) so the page loads in about two minutes.**
Each doubling of n makes the direct method about 8× slower and Strassen's about 7× slower.
At k = 8 both evaluations finish in roughly two minutes; k = 9 takes about 10 minutes,
and k = 10 over an hour.
"""


def is_power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def matrix_to_text(M):
    return "\n".join(" ".join(str(v) for v in row) for row in M) + "\n"


def matrix_editor(label, n, key):
    st.markdown(f"**{label}**")
    df = pd.DataFrame([[1] * n for _ in range(n)], columns=[str(j + 1) for j in range(n)])
    column_config = {
        col: st.column_config.NumberColumn(min_value=1, step=1, required=True)
        for col in df.columns
    }
    edited = st.data_editor(df, key=key, hide_index=True, column_config=column_config)
    if edited.isnull().values.any():
        return None
    return edited.astype(int).values.tolist()


st.set_page_config(page_title="Matrix Multiplication", layout="wide")
st.title("Matrix Multiplication")

with st.sidebar:
    st.header("Algorithms")
    enabled_methods = tuple(
        name for name in PAGE_METHODS if st.toggle(name, value=True, key=f"method_{name}")
    )
    st.header("Evaluations")
    run_eval1 = st.toggle("Evaluation I: operation counts", value=True)
    run_eval2 = st.toggle("Evaluation II: running times", value=True)
    st.caption(f"Evaluates n = 1 … {2 ** EVAL_MAX_K} (k = {EVAL_MAX_K})")
    if st.button("Re-run evaluations"):
        clear_evaluation_caches()

# ---------- Input ----------
st.header("Input")
input_method = st.radio("Input method", ["Enter matrices", "Upload file"], horizontal=True)

n, A, B = None, None, None
if input_method == "Enter matrices":
    n = st.number_input("n", min_value=1, max_value=MAX_EDITOR_N, value=4, step=1)
    left, right = st.columns(2)
    with left:
        A = matrix_editor("A", n, key=f"A_{n}")
    with right:
        B = matrix_editor("B", n, key=f"B_{n}")
    if A is None or B is None:
        st.error("Every entry of A and B must be a positive integer.")
        n = None
else:
    uploaded = st.file_uploader(
        "Text file: n on the first line, then n rows of A, then n rows of B",
        type=["txt"],
    )
    if uploaded is not None:
        try:
            n, A, B = parse_input_text(uploaded.getvalue().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            st.error(f"Could not read file: {e}")
            n = None

# ---------- Product ----------
st.header("Product")
if not enabled_methods:
    st.info("Turn on at least one algorithm in the sidebar.")
elif n is None:
    st.info("Enter matrices or upload a file to compute C = A × B.")
else:
    products = {}
    for method in enabled_methods:
        st.subheader(method)
        if method == "Strassen" and not is_power_of_two(n):
            st.warning(f"Strassen's algorithm requires n to be a power of 2 (got n = {n}).")
            continue
        start = time.perf_counter()
        C = METHODS[method](n, A, B)
        elapsed = time.perf_counter() - start
        products[method] = C
        st.caption(f"Computed in {elapsed * 1000:.3f} ms")
        if n <= MAX_DISPLAY_N:
            st.latex(f"{latex_matrix(C)} = {latex_matrix(A)} \\times {latex_matrix(B)}")
        else:
            st.write(f"n = {n} is too large to display; download the product instead.")
        st.download_button(
            f"Download C ({method})",
            matrix_to_text(C),
            file_name=f"product_{method.lower()}.txt",
            key=f"download_{method}",
        )
    if len(products) == 2:
        if products["Direct"] == products["Strassen"]:
            st.success("Both algorithms produced the same product.")
        else:
            st.error("The algorithms produced different products.")

# ---------- Evaluations ----------
if enabled_methods and (run_eval1 or run_eval2):
    st.header("Performance evaluations")
    st.info(EVAL_CAP_NOTE)

if enabled_methods and run_eval1:
    st.subheader("Evaluation I: operation counts (all-ones matrices)")
    with st.spinner("Counting operations…"):
        evaluation1 = build_evaluation1(EVAL_MAX_K, enabled_methods)
    show_evaluation_tables(evaluation1, lambda method: evaluation1.results[method])
    show_figure(evaluation1.make_figure())

if enabled_methods and run_eval2:
    st.subheader("Evaluation II: wall-clock running times (all-ones matrices)")
    with st.spinner("Timing algorithms…"):
        evaluation2 = build_evaluation2(EVAL_MAX_K, enabled_methods)
    show_evaluation_tables(evaluation2, evaluation2.formatted_rows)
    show_figure(evaluation2.make_figure())
