import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from performance_evaluations import Evaluation1, Evaluation2

EVAL_MAX_K = 8
EVAL_MIN_TIME = 0.5
CROSSOVER_MAX_K = 7
EXTENSION2_MAX_N = 128
HIGHLIGHT = "background-color: rgba(46, 160, 67, 0.3)"


PADDING_MAX_N = 20
ENTRY_COLOR = "#4C78A8"
ZERO_COLOR = "#D9D9D9"
PADDING_LEGEND = [
    Patch(color=ENTRY_COLOR, label="matrix entries"),
    Patch(color=ZERO_COLOR, label="added zeros"),
]


def latex_matrix(M):
    rows = r" \\ ".join(" & ".join(str(v) for v in row) for row in M)
    return r"\begin{pmatrix}" + rows + r"\end{pmatrix}"


def draw_grid(ax, size, entries, title, split=False):
    # size x size grid whose leading entries x entries block is real data, the rest padded zeros
    cells = [[1 if i < entries and j < entries else 0 for j in range(size)] for i in range(size)]
    ax.imshow(cells, cmap=ListedColormap([ZERO_COLOR, ENTRY_COLOR]), vmin=0, vmax=1)
    ax.set_xticks([x - 0.5 for x in range(1, size)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, size)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if split:
        ax.axhline(size / 2 - 0.5, color="black", linewidth=2.5)
        ax.axvline(size / 2 - 0.5, color="black", linewidth=2.5)
    ax.set_title(title, fontsize=10)


def static_padding_figure(n):
    size = 1 << (n - 1).bit_length()
    fig, ax = plt.subplots(figsize=(4, 4))
    draw_grid(ax, size, n, f"{n} × {n} padded to {size} × {size}", split=size > 1)
    ax.legend(
        handles=PADDING_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False
    )
    return fig


def show_static_padding(n):
    size = 1 << (n - 1).bit_length()
    show_figure(static_padding_figure(n))
    zeros = size**2 - n**2
    st.caption(
        f"{zeros} of the {size ** 2} entries ({zeros / size ** 2:.0%}) are added zeros, "
        "and Strassen's algorithm then runs on the full padded matrix."
    )


# Cached per method so Direct/Strassen rows are computed once and shared by both pages.
@st.cache_resource(show_spinner=False)
def evaluation1_crossover(max_k):
    evaluation = Evaluation1(max_k, ("Hybrid",))
    evaluation.estimate_crossover()
    return evaluation.h, evaluation.crossover_rows


@st.cache_resource(show_spinner=False)
def evaluation1_rows(max_k, method, h):
    return Evaluation1(max_k, (method,), h=h).run_method(method)


@st.cache_resource(show_spinner=False)
def evaluation2_crossover(max_k):
    evaluation = Evaluation2(max_k, EVAL_MIN_TIME, ("Hybrid",))
    evaluation.estimate_crossover()
    return evaluation.h, evaluation.crossover_rows


@st.cache_resource(show_spinner=False)
def evaluation2_rows(max_k, method, h):
    return Evaluation2(max_k, EVAL_MIN_TIME, (method,), h=h).run_method(method)


@st.cache_resource(show_spinner=False)
def family_evaluation1_rows(sizes, method):
    return Evaluation1(0, (method,), sizes=sizes).run_method(method)


@st.cache_resource(show_spinner=False)
def family_evaluation2_rows(sizes, method):
    return Evaluation2(0, EVAL_MIN_TIME, (method,), sizes=sizes).run_method(method)


def clear_evaluation_caches():
    for cached in (
        evaluation1_crossover, evaluation1_rows, evaluation2_crossover, evaluation2_rows,
        family_evaluation1_rows, family_evaluation2_rows,
    ):
        cached.clear()


def build_family_evaluation1(sizes, methods, labels=None):
    evaluation = Evaluation1(0, methods, sizes=sizes, labels=labels)
    evaluation.results = {method: family_evaluation1_rows(tuple(sizes), method) for method in methods}
    return evaluation


def build_family_evaluation2(sizes, methods, labels=None):
    evaluation = Evaluation2(0, EVAL_MIN_TIME, methods, sizes=sizes, labels=labels)
    evaluation.results = {method: family_evaluation2_rows(tuple(sizes), method) for method in methods}
    return evaluation


def build_evaluation1(max_k, methods):
    evaluation = Evaluation1(max_k, methods)
    if "Hybrid" in methods:
        evaluation.h, evaluation.crossover_rows = evaluation1_crossover(CROSSOVER_MAX_K)
    evaluation.results = {
        method: evaluation1_rows(max_k, method, evaluation.h if method == "Hybrid" else None)
        for method in methods
    }
    return evaluation


def build_evaluation2(max_k, methods):
    evaluation = Evaluation2(max_k, EVAL_MIN_TIME, methods)
    if "Hybrid" in methods:
        evaluation.h, evaluation.crossover_rows = evaluation2_crossover(CROSSOVER_MAX_K)
    evaluation.results = {
        method: evaluation2_rows(max_k, method, evaluation.h if method == "Hybrid" else None)
        for method in methods
    }
    return evaluation


def show_evaluation_tables(evaluation, rows_for):
    columns = st.columns(len(evaluation.results))
    for column, method in zip(columns, evaluation.results):
        with column:
            st.markdown(f"**{evaluation.label(method)} method**")
            st.dataframe(
                pd.DataFrame(rows_for(method), columns=evaluation.HEADERS),
                hide_index=True,
                width="stretch",
            )


def show_highlighted_table(rows, headers, highlight_index):
    df = pd.DataFrame(rows, columns=headers)
    styled = df.style.apply(
        lambda row: [HIGHLIGHT if row.name == highlight_index else ""] * len(row), axis=1
    )
    st.dataframe(styled, hide_index=True, width="stretch")


def show_h(evaluation):
    h = evaluation.h
    if evaluation.crossover_found():
        st.success(
            f"**h = {h}**: the direct method is faster for n ≤ {2 ** h}, "
            f"and Strassen is faster for n > {2 ** h}."
        )
    else:
        st.warning(
            f"No crossover up to n = {2 ** CROSSOVER_MAX_K}: the direct method was faster at "
            f"every size, so h = {h} and the hybrid uses the direct method throughout."
        )


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)
