import streamlit as st

from performance_evaluations import ALL_METHODS, Evaluation1
from ui_helpers import (
    CROSSOVER_MAX_K,
    EVAL_MAX_K,
    EVAL_MIN_TIME,
    build_evaluation1,
    build_evaluation2,
    show_evaluation_tables,
    show_figure,
    show_h,
    show_highlighted_table,
)

st.set_page_config(page_title="Extension I: Hybrid", layout="wide")
st.title("Extension I: Hybrid algorithm")
st.markdown(
    "The hybrid algorithm follows Strassen's recursion until the submatrices are "
    "2^h × 2^h, then multiplies them with the direct method. Each evaluation method "
    "estimates its own crossover threshold h, and its hybrid uses that h."
)
st.caption(
    f"All results use all-ones matrices for n = 1 … {2 ** EVAL_MAX_K} (k = {EVAL_MAX_K}), "
    "the same cap as the main page."
)

with st.spinner("Running Evaluation I (operation counts)…"):
    evaluation1 = build_evaluation1(EVAL_MAX_K, ALL_METHODS)
with st.spinner("Running Evaluation II (running times)…"):
    evaluation2 = build_evaluation2(EVAL_MAX_K, ALL_METHODS)

# ---------- How h was estimated ----------
st.header("Estimating h")
st.markdown(
    f"""
For each n = 2^i up to {2 ** CROSSOVER_MAX_K}, we multiply n × n all-ones matrices two ways:

- **Direct**: the direct method on the whole matrix.
- **Strassen**: one level of Strassen, with the seven (n/2) × (n/2) products computed by
  the direct method.

The highlighted row is the first n where Strassen is faster, so **h = i − 1**:
the direct method is faster for n ≤ 2^h and Strassen for n > 2^h.
"""
)

st.subheader("Evaluation I: operation counts")
direct_rows = [row[0:4] for row in evaluation1.crossover_rows]
strassen_rows = [(row[0],) + row[4:7] for row in evaluation1.crossover_rows]
left, right = st.columns(2)
with left:
    st.markdown("**Direct**")
    show_highlighted_table(direct_rows, Evaluation1.HEADERS, evaluation1.crossover_index())
with right:
    st.markdown("**Strassen**")
    show_highlighted_table(strassen_rows, Evaluation1.HEADERS, evaluation1.crossover_index())
show_h(evaluation1)

st.subheader("Evaluation II: running time")
st.markdown(
    f"Average running time over repeated runs lasting at least {EVAL_MIN_TIME} s."
)
show_highlighted_table(
    [row[0:3] for row in evaluation2.crossover_rows],
    ("n", "Direct", "Strassen (one split, then direct)"),
    evaluation2.crossover_index(),
)
show_h(evaluation2)
st.caption("Measured times vary between runs, so on another run or machine this h can shift by one.")

# ---------- Comparison tables and charts ----------
st.header("Operation counts")
show_evaluation_tables(evaluation1, lambda method: evaluation1.results[method])
show_figure(evaluation1.make_figure())

st.header("Running times")
show_evaluation_tables(evaluation2, evaluation2.formatted_rows)
show_figure(evaluation2.make_figure())
