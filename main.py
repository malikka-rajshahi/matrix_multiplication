import argparse
import os
import sys

from matrix_algorithms import MatrixMultiplication, StrassensAlgorithm
from performance_evaluations import (
    EXTENSION2_LABELS,
    EXTENSION2_METHODS,
    PADDING_METHODS,
    Evaluation1,
    Evaluation2,
    correctness_examples,
    extension2_sizes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matrix multiplication driver."
    )
    parser.add_argument(
        "-u", "--ui",
        action="store_true",
        help="launch the web UI in a browser",
    )
    parser.add_argument(
        "-f", "--file",
        metavar="PATH",
        help="path to a file containing the matrices",
    )
    parser.add_argument(
        "-e1", "--evaluation1",
        action="store_true",
        help="run performance evaluation I (operation counts)",
    )
    parser.add_argument(
        "-e2", "--evaluation2",
        action="store_true",
        help="run performance evaluation II (wall-clock running times)",
    )
    parser.add_argument(
        "-x2", "--extension2",
        action="store_true",
        help="run extension II (direct vs. Strassen for any n: correctness examples and "
             "both evaluations per family of n)",
    )
    parser.add_argument(
        "-x2a", "--extension2-additional",
        action="store_true",
        help="run the extension II additional part (static vs. dynamic padding per family of n)",
    )
    parser.add_argument(
        "-n", "--max-n",
        type=int,
        default=128,
        help="extension II: largest n in each family (default: 128)",
    )
    parser.add_argument(
        "-t", "--min-time",
        type=float,
        default=0.5,
        help="evaluation II: repeat each run until it takes at least this many seconds (default: 0.5)",
    )
    parser.add_argument(
        "-k", "--max-k",
        type=int,
        default=7,
        help="evaluate n = 2^0 .. 2^k (default: 7)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="charts",
        help="directory for chart images (default: charts)",
    )
    args = parser.parse_args()

    if args.ui and args.file is not None:
        parser.error(
            "Input can only be done through the interface or by providing a filepath"
        )
    running_evaluations = (
        args.evaluation1 or args.evaluation2 or args.extension2 or args.extension2_additional
    )
    if running_evaluations and (args.interface or args.file is not None):
        parser.error("evaluations cannot be combined with --interface or --file")

    return args


def run_evaluation1(max_k, output_dir):
    evaluation = Evaluation1(max_k)
    evaluation.run()
    evaluation.print_crossover()
    evaluation.print_tables()
    print(f"Saved {evaluation.plot_charts(output_dir)}")


def run_evaluation2(max_k, min_time, output_dir):
    evaluation = Evaluation2(max_k, min_time)
    evaluation.run()
    evaluation.print_crossover()
    evaluation.print_tables()
    print(f"Saved {evaluation.plot_charts(output_dir)}")


def run_family_evaluations(methods, labels, max_n, min_time, output_dir, prefix):
    for number, (family, sizes) in enumerate(extension2_sizes(max_n).items(), start=1):
        print(f"\n===== Family {number}, {family}: n = {sizes} =====")
        slug = f"{prefix}_family{number}"
        evaluation1 = Evaluation1(0, methods, sizes=sizes, labels=labels)
        evaluation1.run()
        evaluation1.print_tables()
        print(f"Saved {evaluation1.plot_charts(output_dir, prefix=f'{slug}_eval1')}")
        evaluation2 = Evaluation2(0, min_time, methods, sizes=sizes, labels=labels)
        evaluation2.run()
        evaluation2.print_tables()
        print(f"Saved {evaluation2.plot_charts(output_dir, prefix=f'{slug}_eval2')}")


def run_extension2(max_n, min_time, output_dir):
    print("Correctness examples")
    for n, A, B, products in correctness_examples():
        print(f"\nn = {n}")
        for name, M in [("A", A), ("B", B)] + list(products.items()):
            print(f"{EXTENSION2_LABELS.get(name, name)}:")
            for row in M:
                print("  " + " ".join(f"{v:4}" for v in row))
        same = all(C == products["Direct"] for C in products.values())
        print("Both methods agree." if same else "Methods DISAGREE.")
    run_family_evaluations(EXTENSION2_METHODS, EXTENSION2_LABELS, max_n, min_time, output_dir, "ext2")


def run_extension2_additional(max_n, min_time, output_dir):
    run_family_evaluations(PADDING_METHODS, None, max_n, min_time, output_dir, "ext2_padding")


def parse_input_file(file_path):
    with open(file_path, "r") as f:
        return parse_input_text(f.read())


def parse_input_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        raise ValueError("Input file is empty.")

    # Parse first line to get n
    first_line = lines[0].split()
    if not first_line:
        raise ValueError("The first line in the input file is empty.")
    n = int(first_line[-1])  # Take last value in case of extra columns

    expected_lines = 2 * n + 1
    if len(lines) < expected_lines:
        raise ValueError(
            f"Expected {expected_lines} non-empty lines for n={n} "
            f"(1 header + {n} rows A + {n} rows B), but found {len(lines)}."
        )

    # Parse matrix A
    A = []
    for i in range(1, n + 1):
        row = list(map(int, lines[i].split()))
        if len(row) < n:
            raise ValueError(
                f"Matrix A row {i} has {len(row)} values; expected at least {n}."
            )
        A.append(row[-n:])  # Take last n values in case of extra columns

    # Parse matrix B (next n lines)
    B = []
    for i in range(n + 1, 2 * n + 1):
        row = list(map(int, lines[i].split()))
        if len(row) < n:
            raise ValueError(
                f"Matrix B row {i - n} has {len(row)} values; expected at least {n}."
            )
        B.append(row[-n:])  # Take last n values in case of extra columns

    return n, A, B

if __name__ == "__main__":
    args = parse_args()

    if args.ui:
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.py")
        # replace this process with Streamlit so Ctrl+C stops it without a traceback from here
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", ui_path])

    if args.evaluation1 or args.evaluation2 or args.extension2 or args.extension2_additional:
        if args.evaluation1:
            run_evaluation1(args.max_k, args.output_dir)
        if args.evaluation2:
            run_evaluation2(args.max_k, args.min_time, args.output_dir)
        if args.extension2:
            run_extension2(args.max_n, args.min_time, args.output_dir)
        if args.extension2_additional:
            run_extension2_additional(args.max_n, args.min_time, args.output_dir)
        raise SystemExit(0)

    n = 0
    A = [[]]
    B = [[]]

    if args.file is not None:
        try:
            n, A, B = parse_input_file(args.file)
        except ValueError as e:
            raise SystemExit(str(e))
        
    mm_C = MatrixMultiplication.matrix_multiplication(n, A, B)
    try:
        strassen_C = StrassensAlgorithm.strassens_algorithm(n, A, B)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.file is not None:
        print(f"n = {n}")
        for name, M in [("A", A), ("B", B), ("Direct", mm_C), ("Strassen", strassen_C)]:
            print(f"{name}:")
            for row in M:
                print("  " + " ".join(f"{v:4}" for v in row))
        print("Both methods agree." if mm_C == strassen_C else "Methods DISAGREE.")
    


