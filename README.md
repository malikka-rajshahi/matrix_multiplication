# Matrix Multiplication: Direct Method vs. Strassen's Algorithm

This project multiplies square matrices with the direct (schoolbook) method and with Strassen's algorithm. It compares them by operation counts and by running time, and it extends Strassen's algorithm to any n by padding the matrices with zeros. You can run it from the command line or in an interactive web UI built with Streamlit.

## Project layout

| File / folder | Contents |
|---|---|
| `main.py` | Command-line entry point: multiplies the matrices in an input file, runs the evaluations, or launches the UI |
| `matrix_algorithms.py` | The direct method and Strassen's algorithm |
| `performance_evaluations.py` | Evaluation I (operation counts), Evaluation II (running times) and the Extension II experiments |
| `ui.py`, `ui_helpers.py` | The main page of the web UI and the helpers it shares with the other pages |
| `pages/` | The other UI pages: Extension I, Extension II and Extension II Additional |
| `input.txt` | A sample input file |
| `charts/` | Charts the command-line evaluations save |

## Environment setup

You need Python 3.12 or newer. Check your version:

```bash
python3 --version
```

From the project folder, create a virtual environment and install the dependencies (Streamlit and Matplotlib):

```bash
cd matrix_multiplication
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Activate the environment (`source .venv/bin/activate`) in every new terminal before you run the code. Run `deactivate` to leave it.

## Running the code

Run every command from the project folder with the virtual environment active. You can run the code in a web UI or in the terminal.

### Option 1: Web UI

```bash
python main.py --ui
```

This command does the same thing as `streamlit run ui.py`. Streamlit opens the app in your browser at http://localhost:8501. If the browser doesn't open, go to that address yourself. Press Ctrl+C in the terminal to stop the server.

Use the sidebar to switch between pages:

- **Matrix Multiplication** (main page): enter matrices or upload an input file, multiply them with both methods, and see Evaluations I and II.
- **Extension I**: the hybrid algorithm.
- **Extension II**: the direct method vs. Strassen's algorithm for any n.
- **Extension II Additional**: static vs. dynamic padding.

Each page can take about a minute to load the first time while it runs its evaluations.

### Option 2: Terminal

#### Multiply the matrices in a file

```bash
python main.py -f input.txt
```

This prints A, B, the product from the direct method, the product from Strassen's algorithm, and whether the two products agree. The input file's first line holds n, which must be a power of 2 (1, 2, 4, 8, ...). The next n lines are the rows of A, and the n lines after them are the rows of B. Separate the values with spaces:

```
2
1 2
3 4
5 6
7 8
```

If n is not a power of 2, the command stops with an error. For those sizes, the main page of the web UI still shows the product from the direct method, and the Extension II page shows padded Strassen on example sizes n = 3, 5, 7 and 9.

#### Performance evaluations

```bash
python main.py -e1     # Evaluation I: operation counts
python main.py -e2     # Evaluation II: running times
python main.py -x2     # Extension II: direct vs. Strassen for any n
python main.py -x2a    # Extension II Additional: static vs. dynamic padding
```

You can combine the flags, for example `python main.py -e1 -e2`. Results print to the terminal as tables, and the charts are saved to `charts/`. You can't combine an evaluation flag with `-f`.

These options change the defaults:

| Option | What it sets | Default |
|---|---|---|
| `-k`, `--max-k` | Sizes evaluated: n = 2^0 up to 2^k | 7 |
| `-n`, `--max-n` | Largest n in each family for Extension II | 128 |
| `-t`, `--min-time` | Minimum seconds per timing run; each run repeats until it lasts at least this long | 0.5 |
| `-o`, `--output-dir` | Folder for the chart images | `charts` |

For example, `python main.py -e2 -k 8 -t 1` times sizes up to n = 256 and makes each timing run last at least 1 second. Larger values of `-k`, `-n` and `-t` make the runs take longer. For a quick check, use small values: `python main.py -x2 -n 8 -t 0.05`.

Run `python main.py -h` to see every option.
