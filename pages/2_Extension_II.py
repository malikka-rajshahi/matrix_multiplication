import streamlit as st

from performance_evaluations import (
    EXTENSION2_LABELS,
    EXTENSION2_METHODS,
    correctness_examples,
    extension2_sizes,
)
from ui_helpers import (
    EVAL_MIN_TIME,
    EXTENSION2_MAX_N,
    PADDING_MAX_N,
    build_family_evaluation1,
    build_family_evaluation2,
    latex_matrix,
    show_evaluation_tables,
    show_figure,
    show_static_padding,
)

st.set_page_config(page_title="Extension II: Any n", layout="wide")
st.title("Extension II: Any n")
st.markdown(
    """
The direct method already works for any n ≥ 1. Strassen's algorithm splits the matrices in half
at every level, so on its own it needs n to be a power of 2. For any other n we use the standard
approach: pad both matrices with zeros to the smallest 2^k ≥ n, run Strassen's algorithm, and
return the leading n × n block of the result (**static padding**). Padding this way keeps
Strassen's Θ(n^log₂7) bound. An alternative that pads at each recursive step is compared on the
Extension II Additional page.
"""
)

# ---------- Static padding ----------
st.header("Static padding")
padding_n = st.number_input(
    "n", min_value=1, max_value=PADDING_MAX_N, value=5, step=1, key="static_padding_n"
)
figure_column, _ = st.columns(2)
with figure_column:
    show_static_padding(padding_n)

# ---------- Correctness ----------
st.header("Correctness: n = 3, 5, 7, 9")
st.markdown(
    "Each example multiplies two random matrices with entries from 1 to 9 "
    "(the same examples every time) using both methods."
)
examples = correctness_examples()
for tab, (n, A, B, products) in zip(st.tabs([f"n = {n}" for n, *_ in examples]), examples):
    with tab:
        st.latex(f"A = {latex_matrix(A)} \\qquad B = {latex_matrix(B)}")
        for method, C in products.items():
            st.markdown(f"**{EXTENSION2_LABELS.get(method, method)}**")
            st.latex(f"C = {latex_matrix(C)}")
        if all(C == products["Direct"] for C in products.values()):
            st.success("Both methods produce the same product.")
        else:
            st.error("The methods produce different products.")

# ---------- Performance ----------
st.header("Performance")
st.markdown(
    f"""
Both evaluations use all-ones matrices, as on the main page. Each family runs up to
n = {EXTENSION2_MAX_N}. A larger n just above a power of 2 (such as 129) makes static padding run
Strassen's algorithm on 256 × 256 matrices, which takes about 18 seconds per run.
"""
)

families = extension2_sizes(EXTENSION2_MAX_N)
for tab, (family, sizes) in zip(st.tabs(list(families)), families.items()):
    with tab:
        st.caption(f"n = {', '.join(str(n) for n in sizes)}")

        st.subheader("Evaluation I: operation counts")
        with st.spinner("Counting operations…"):
            evaluation1 = build_family_evaluation1(sizes, EXTENSION2_METHODS, EXTENSION2_LABELS)
        show_evaluation_tables(evaluation1, lambda method: evaluation1.results[method])
        show_figure(evaluation1.make_figure())

        st.subheader("Evaluation II: running times")
        st.caption(f"Average over repeated runs lasting at least {EVAL_MIN_TIME} s.")
        with st.spinner("Timing algorithms…"):
            evaluation2 = build_family_evaluation2(sizes, EXTENSION2_METHODS, EXTENSION2_LABELS)
        show_evaluation_tables(evaluation2, evaluation2.formatted_rows)
        show_figure(evaluation2.make_figure())
