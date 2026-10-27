import math

import matplotlib.pyplot as plt
import streamlit as st

from performance_evaluations import PADDING_METHODS, Evaluation1, extension2_sizes
from ui_helpers import (
    EVAL_MIN_TIME,
    EXTENSION2_MAX_N,
    PADDING_LEGEND,
    PADDING_MAX_N,
    build_family_evaluation1,
    build_family_evaluation2,
    draw_grid,
    show_evaluation_tables,
    show_figure,
    show_static_padding,
)


def dynamic_padding_steps(n):
    # one branch of the recursion: pad odd sizes by one, halve even sizes, stop at 1
    steps = []
    while n > 1:
        if n % 2 == 1:
            steps.append(("pad", n, n + 1))
            n += 1
        else:
            steps.append(("split", n, n // 2))
            n //= 2
    return steps


def dynamic_padding_figure(n):
    steps = dynamic_padding_steps(n)
    panels = len(steps) + 1
    columns = min(panels, 4)
    rows = math.ceil(panels / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.6 * rows), squeeze=False)
    axes = [ax for row in axes for ax in row]
    just_padded = False
    for ax, (action, size, next_size) in zip(axes, steps):
        if action == "pad":
            draw_grid(ax, next_size, size, f"{size}: pad to {next_size}")
        else:
            # a split right after a pad still carries that pad's zero row and column
            entries = size - 1 if just_padded else size
            draw_grid(ax, size, entries, f"{size}: split into {next_size}s", split=True)
        just_padded = action == "pad"
    draw_grid(axes[len(steps)], 1, 1, "1: multiply")
    for ax in axes[panels:]:
        ax.axis("off")
    fig.legend(handles=PADDING_LEGEND, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2)
    return fig


st.set_page_config(page_title="Extension II: Additional", layout="wide")
st.title("Extension II: Additional")
st.markdown(
    """
Two ways to run Strassen's algorithm when n is not a power of 2:

- **Static padding**: pad both matrices with zeros once, to the smallest 2^k ≥ n, run Strassen's
  algorithm, and return the leading n × n block of the result.
- **Dynamic padding**: at each recursive step, split normally when n is even. When n is odd, first
  add a zero row and column, recurse on the (n + 1) × (n + 1) matrices, and drop the added row and
  column from the result.
"""
)

# ---------- Padding ----------
st.header("Padding")
padding_n = st.number_input(
    "n", min_value=1, max_value=PADDING_MAX_N, value=5, step=1, key="padding_n"
)
metrics = st.columns(len(PADDING_METHODS))
for column, method in zip(metrics, PADDING_METHODS):
    column.metric(f"{method}: total operations", f"{sum(Evaluation1.theoretical_counts(method, padding_n)):,}")

left, right = st.columns(2)
with left:
    st.subheader("Static padding")
    show_static_padding(padding_n)
with right:
    st.subheader("Dynamic padding")
    steps = dynamic_padding_steps(padding_n)
    st.markdown(" → ".join([str(padding_n)] + [str(next_size) for _, _, next_size in steps]))
    show_figure(dynamic_padding_figure(padding_n))
    st.caption(
        "One branch of the recursion: every split creates 7 subproblems of half the size, and each "
        "pads only by a single row and column when its size is odd."
    )

# ---------- Comparison ----------
st.header("Static vs. dynamic padding")
st.markdown(
    f"""
Both evaluations use all-ones matrices for the same families of n as the Extension II page,
up to n = {EXTENSION2_MAX_N}.
"""
)

families = extension2_sizes(EXTENSION2_MAX_N)
for tab, (family, sizes) in zip(st.tabs(list(families)), families.items()):
    with tab:
        st.caption(f"n = {', '.join(str(n) for n in sizes)}")

        st.subheader("Evaluation I: operation counts")
        with st.spinner("Counting operations…"):
            evaluation1 = build_family_evaluation1(sizes, PADDING_METHODS)
        show_evaluation_tables(evaluation1, lambda method: evaluation1.results[method])
        show_figure(evaluation1.make_figure())

        st.subheader("Evaluation II: running times")
        st.caption(f"Average over repeated runs lasting at least {EVAL_MIN_TIME} s.")
        with st.spinner("Timing algorithms…"):
            evaluation2 = build_family_evaluation2(sizes, PADDING_METHODS)
        show_evaluation_tables(evaluation2, evaluation2.formatted_rows)
        show_figure(evaluation2.make_figure())
