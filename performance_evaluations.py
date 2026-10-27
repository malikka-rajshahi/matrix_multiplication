import os
import random
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matrix_algorithms import (
    DynamicPaddingStrassen,
    HybridAlgorithm,
    MatrixMultiplication,
    StaticPaddingStrassen,
    StrassensAlgorithm,
)


METHODS = {
    "Direct": MatrixMultiplication.matrix_multiplication,
    "Strassen": StrassensAlgorithm.strassens_algorithm,
    "Static padding": StaticPaddingStrassen.static_padding_strassen,
    "Dynamic padding": DynamicPaddingStrassen.dynamic_padding_strassen,
}
ALL_METHODS = ("Direct", "Strassen", "Hybrid")
# CountedInt's counters are shared and concurrent work skews timings, so when several threads
# evaluate at once (e.g. Streamlit sessions), only one count or timing runs at a time
MEASUREMENT_LOCK = threading.Lock()
# Extension II compares direct with Strassen for any n, which pads to the next 2^k (static padding);
# the additional extension compares the two padding methods
EXTENSION2_METHODS = ("Direct", "Static padding")
EXTENSION2_LABELS = {"Static padding": "Strassen (static padding)"}
PADDING_METHODS = ("Static padding", "Dynamic padding")
EXTENSION2_FAMILIES = {
    "n = 2^k + 1": lambda k: 2**k + 1,
    "n = 2^(k+1) - 1": lambda k: 2 ** (k + 1) - 1,
    "n = 3 * 2^(k-1)": lambda k: 3 * 2 ** (k - 1),
}


def extension2_sizes(max_n):
    # each family's n values for k >= 1, up to max_n
    sizes = {}
    for name, n_of_k in EXTENSION2_FAMILIES.items():
        sizes[name] = []
        k = 1
        while n_of_k(k) <= max_n:
            sizes[name].append(n_of_k(k))
            k += 1
    return sizes


def correctness_examples(methods=EXTENSION2_METHODS, sizes=(3, 5, 7, 9), seed=7010):
    # random positive-integer matrices (fixed seed) multiplied by each method
    rng = random.Random(seed)
    examples = []
    for n in sizes:
        A = [[rng.randint(1, 9) for _ in range(n)] for _ in range(n)]
        B = [[rng.randint(1, 9) for _ in range(n)] for _ in range(n)]
        products = {method: METHODS[method](n, A, B) for method in methods}
        examples.append((n, A, B, products))
    return examples


def print_table(title, headers, rows):
    cells = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[c]), *(len(row[c]) for row in cells))
        for c in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(f"\n{title}")
    print(separator)
    print("| " + " | ".join(h.center(w) for h, w in zip(headers, widths)) + " |")
    print(separator)
    for row in cells:
        print("| " + " | ".join(v.rjust(w) for v, w in zip(row, widths)) + " |")
    print(separator)


def save_figure(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def set_n_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    # label the actual n values when they aren't all powers of 2
    if any(n & (n - 1) for n in sizes):
        ax.set_xticks(sizes, labels=[str(n) for n in sizes])
        ax.minorticks_off()


def one_split(k):
    # hybrid that does exactly one Strassen split at n = 2^k, then multiplies the halves directly
    return lambda n, A, B: HybridAlgorithm.hybrid_algorithm(n, A, B, k - 1)


class CountedInt:
    """Integer that counts every addition/subtraction and multiplication it takes part in."""

    __slots__ = ("value",)
    additions = 0
    multiplications = 0

    def __init__(self, value):
        self.value = value

    @classmethod
    def reset(cls):
        cls.additions = 0
        cls.multiplications = 0

    def __add__(self, other):
        CountedInt.additions += 1
        return CountedInt(self.value + other.value)

    def __sub__(self, other):
        CountedInt.additions += 1
        return CountedInt(self.value - other.value)

    def __mul__(self, other):
        CountedInt.multiplications += 1
        return CountedInt(self.value * other.value)


class Evaluation:
    """Shared by both evaluations: method lookup, hybrid threshold h, and crossover bookkeeping."""

    def __init__(self, max_k, methods=ALL_METHODS, h=None, sizes=None, labels=None):
        self.max_k = max_k
        self.sizes = list(sizes) if sizes else [2**k for k in range(max_k + 1)]
        self.method_names = tuple(methods)
        self.h = h
        self.labels = labels or {}
        self.crossover_rows = []
        self.results = {}

    def algorithm(self, method):
        if method == "Hybrid":
            h = self.h
            return lambda n, A, B: HybridAlgorithm.hybrid_algorithm(n, A, B, h)
        return METHODS[method]

    def label(self, method):
        if method == "Hybrid":
            return f"Hybrid (h = {self.h})"
        return self.labels.get(method, method)

    def crossover_index(self):
        # index of the first row where the Strassen split beats direct, or None
        for i, row in enumerate(self.crossover_rows):
            if row[-1] == "split":
                return i
        return None

    def crossover_found(self):
        return self.crossover_index() is not None

    def estimate_crossover(self):
        # h = i - 1 for the first n = 2^i where one Strassen split beats direct
        self.crossover_rows = [self.crossover_row(k) for k in range(1, self.max_k + 1)]
        index = self.crossover_index()
        self.h = self.max_k if index is None else index
        return self.h

    def run(self):
        if "Hybrid" in self.method_names and self.h is None:
            self.estimate_crossover()
        for method in self.method_names:
            self.results[method] = self.run_method(method)
        return self.results

    def print_crossover(self):
        print_table("Crossover estimate", self.CROSSOVER_HEADERS, self.crossover_rows)
        if self.crossover_found():
            print(f"h = {self.h}: direct is faster for n <= {2 ** self.h}, "
                  f"a Strassen split is faster for n > {2 ** self.h}")
        else:
            print(f"No crossover up to n = {2 ** self.max_k}; h = {self.h}")


class Evaluation1(Evaluation):
    HEADERS = ("n", "#additions/subtractions", "#multiplications", "#operations")
    CROSSOVER_HEADERS = (
        "n",
        "direct #additions/subtractions", "direct #multiplications", "direct #operations",
        "split #additions/subtractions", "split #multiplications", "split #operations",
        "fewer operations",
    )

    @staticmethod
    def dynamic_padding_counts(n):
        # A(1) = 0,  A(n) = A(n + 1) for odd n,  A(n) = 7A(n/2) + 18(n^2/4) for even n
        # M(1) = 1,  M(n) = M(n + 1) for odd n,  M(n) = 7M(n/2)            for even n
        if n == 1:
            return 0, 1
        if n % 2 == 1:
            return Evaluation1.dynamic_padding_counts(n + 1)
        additions, multiplications = Evaluation1.dynamic_padding_counts(n // 2)
        return 7 * additions + 18 * (n // 2) ** 2, 7 * multiplications

    @staticmethod
    def theoretical_counts(method, n, h=None):
        # Direct: n^2(n - 1) additions, n^3 multiplications
        if method == "Direct":
            return n**2 * (n - 1), n**3
        # Static padding: Strassen on the smallest 2^k >= n
        if method == "Static padding":
            return Evaluation1.theoretical_counts("Strassen", 1 << (n - 1).bit_length())
        if method == "Dynamic padding":
            return Evaluation1.dynamic_padding_counts(n)
        k = n.bit_length() - 1
        if method == "Hybrid" and k <= h:
            return n**2 * (n - 1), n**3
        # Strassen (n = 2^k, so 7^log2(n) = 7^k):
        #   A(1) = 0,  A(n) = 7A(n/2) + 18(n^2/4)  ->  A(n) = 6 * 7^log2(n) - 6 * n^2
        #   M(1) = 1,  M(n) = 7M(n/2)              ->  M(n) = 7^log2(n) = n^log2(7)
        if method == "Strassen":
            return 6 * 7**k - 6 * n**2, 7**k
        # Hybrid (base case s = 2^h uses the direct method, j = k - h Strassen levels above it):
        #   A(s) = s^2(s - 1),  A(n) = 7A(n/2) + 18(n^2/4)  ->  A(n) = 7^j * A(s) + ((7/4)^j - 1) * 6 * n^2
        #   M(s) = s^3,         M(n) = 7M(n/2)              ->  M(n) = 7^j * M(s)
        # (7/4)^j * n^2 = 7^j * s^2, which keeps the additions formula in integers
        s, j = 2**h, k - h
        return 7**j * s**2 * (s - 1) + 6 * (7**j * s**2 - n**2), 7**j * s**3

    @staticmethod
    def count_operations(algorithm, n):
        A = [[CountedInt(1) for _ in range(n)] for _ in range(n)]
        B = [[CountedInt(1) for _ in range(n)] for _ in range(n)]
        with MEASUREMENT_LOCK:
            CountedInt.reset()
            C = algorithm(n, A, B)
            counts = CountedInt.additions, CountedInt.multiplications
        if any(entry.value != n for row in C for entry in row):
            raise AssertionError(f"Incorrect product for n={n}")
        return counts

    def crossover_row(self, k):
        # At n = 2^k compare direct, D(n) = 2n^3 - n^2, against one Strassen split with direct
        # on the halves, S(n) = 7D(n/2) + 18(n/2)^2 = (7/4)n^3 + (11/4)n^2.
        # S(n) < D(n)  <->  n^3/4 > (15/4)n^2  <->  n > 15, so the first win is at n = 16 and h = 3.
        n = 2 ** k
        direct_adds, direct_mults = self.count_operations(
            MatrixMultiplication.matrix_multiplication, n
        )
        split_adds, split_mults = self.count_operations(one_split(k), n)
        direct = direct_adds + direct_mults
        split = split_adds + split_mults
        winner = "split" if split < direct else "direct"
        return (n, direct_adds, direct_mults, direct, split_adds, split_mults, split, winner)

    def run_method(self, method):
        algorithm = self.algorithm(method)
        rows = []
        for n in self.sizes:
            additions, multiplications = self.count_operations(algorithm, n)
            expected = self.theoretical_counts(method, n, self.h)
            if (additions, multiplications) != expected:
                raise AssertionError(
                    f"{method}, n={n}: counted {(additions, multiplications)}, "
                    f"expected {expected}"
                )
            rows.append((n, additions, multiplications, additions + multiplications))
        return rows

    def print_tables(self):
        for method, rows in self.results.items():
            print_table(f"{self.label(method)} method", self.HEADERS, rows)

    def make_figure(self):
        charts = [
            (1, "Additions/subtractions"),
            (2, "Multiplications"),
            (3, "Total operations"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        for ax, (column, title) in zip(axes, charts):
            for method, rows in self.results.items():
                # log scale can't show 0, so skip n=1 where direct/Strassen do 0 additions
                points = [(row[0], row[column]) for row in rows if row[column] > 0]
                ax.plot(*zip(*points), marker="o", label=self.label(method))
            set_n_axis(ax, self.sizes)
            ax.set_yscale("log")
            ax.set_xlabel("n")
            ax.set_ylabel("Count")
            ax.set_title(f"{title} vs. n")
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            ax.legend()
        fig.tight_layout()
        return fig

    def plot_charts(self, output_dir, prefix="eval1"):
        return save_figure(self.make_figure(), output_dir, f"{prefix}_operation_counts.png")


class Evaluation2(Evaluation):
    HEADERS = ("n", "m (repetitions)", "total time (s)", "average time (s)")
    CROSSOVER_HEADERS = ("n", "direct", "one Strassen split + direct", "faster")

    def __init__(self, max_k, min_time, methods=ALL_METHODS, h=None, sizes=None, labels=None):
        super().__init__(max_k, methods, h, sizes, labels)
        self.min_time = min_time

    def measure_time(self, algorithm, n):
        A = [[1] * n for _ in range(n)]
        B = [[1] * n for _ in range(n)]
        # double m until m runs take at least min_time, so short runs are still measurable
        m = 1
        with MEASUREMENT_LOCK:
            while True:
                start = time.perf_counter()
                for _ in range(m):
                    algorithm(n, A, B)
                total = time.perf_counter() - start
                if total >= self.min_time:
                    return m, total, total / m
                m *= 2

    def crossover_row(self, k):
        # same comparison as Evaluation1, but with measured average running times
        n = 2 ** k
        direct = self.measure_time(MatrixMultiplication.matrix_multiplication, n)[2]
        split = self.measure_time(one_split(k), n)[2]
        winner = "split" if split < direct else "direct"
        return (n, f"{direct:.6e} s", f"{split:.6e} s", winner)

    def run_method(self, method):
        algorithm = self.algorithm(method)
        rows = []
        for n in self.sizes:
            m, total, average = self.measure_time(algorithm, n)
            rows.append((n, m, total, average))
        return rows

    def formatted_rows(self, method):
        return [
            (n, m, f"{total:.4f}", f"{average:.6e}")
            for n, m, total, average in self.results[method]
        ]

    def print_tables(self):
        for method in self.results:
            print_table(f"{self.label(method)} method", self.HEADERS, self.formatted_rows(method))

    def make_figure(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, rows in self.results.items():
            ax.plot(
                [row[0] for row in rows], [row[3] for row in rows],
                marker="o", label=self.label(method),
            )
        set_n_axis(ax, self.sizes)
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("Average running time (s)")
        ax.set_title("Wall-clock running time vs. n")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_charts(self, output_dir, prefix="eval2"):
        return save_figure(self.make_figure(), output_dir, f"{prefix}_running_times.png")
