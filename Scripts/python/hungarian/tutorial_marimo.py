import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import inspect
    import math
    import sys
    import time
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")

    notebook_dir = Path(__file__).resolve().parent
    if str(notebook_dir) not in sys.path:
        sys.path.append(str(notebook_dir))

    from algorithm import (
        brute_force_assignment,
        certificate_summary,
        greedy_row_assignment,
        hungarian,
        initial_dual_from_reductions,
        row_reduction,
    )
    from synthetic_cases import default_case_suite
    from visualization import (
        plot_bipartite_assignment,
        plot_cost_heatmap,
        plot_spatial_assignment,
        plot_trace_frame,
    )

    return (
        brute_force_assignment,
        certificate_summary,
        dedent,
        default_case_suite,
        greedy_row_assignment,
        hungarian,
        initial_dual_from_reductions,
        inspect,
        math,
        mo,
        np,
        plot_bipartite_assignment,
        plot_cost_heatmap,
        plot_spatial_assignment,
        plot_trace_frame,
        plt,
        row_reduction,
        time,
    )


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    # Hungarian Algorithm, Really Explained

    This notebook builds the Hungarian algorithm from first principles:

    - the **assignment problem** and why brute force explodes,
    - the **row/column reduction trick** and why it is mathematically valid,
    - the **dual variables** behind the algorithm,
    - a **full Python implementation** with trace capture,
    - **animations, heatmaps, bipartite graphs, spatial plots, benchmarks, and stress tests**.

    Everything here is pure Python and lives in the same folder as this notebook.
    """).strip())
    return


@app.cell
def _(default_case_suite, mo):
    case_selector = mo.ui.dropdown(
        options=list(default_case_suite(seed=7).keys()),
        value="Classic 3x3",
        label="Synthetic case",
        full_width=True,
    )
    seed_slider = mo.ui.slider(
        start=0,
        stop=30,
        step=1,
        value=7,
        label="Seed for generated cases",
        show_value=True,
        full_width=True,
    )
    return case_selector, seed_slider


@app.cell
def _(case_selector, mo, seed_slider):
    mo.vstack([case_selector, seed_slider])
    return


@app.cell
def _(default_case_suite, seed_slider):
    cases = default_case_suite(seed=seed_slider.value)
    return (cases,)


@app.cell
def _(
    brute_force_assignment,
    case_selector,
    cases,
    certificate_summary,
    greedy_row_assignment,
    hungarian,
):
    case = cases[case_selector.value]
    result = hungarian(case.cost, record_trace=True)
    certificate = certificate_summary(result)

    brute = None
    if max(case.cost.shape) <= 8:
        brute = brute_force_assignment(case.cost)

    greedy = None
    if case.cost.shape[0] <= case.cost.shape[1]:
        greedy_assignment, greedy_total = greedy_row_assignment(case.cost)
        greedy = {"assignment": greedy_assignment, "total": greedy_total}

    matched_lines = []
    for row, col in result.assignment:
        matched_lines.append(
            f"- `{case.row_labels[row]}` -> `{case.col_labels[col]}` with cost `{case.cost[row, col]:.2f}`"
        )
    unmatched = [case.row_labels[row] for row, col in enumerate(result.row_to_col) if col == -1]

    summary_table = [
        "| quantity | value |",
        "| --- | --- |",
        f"| matrix shape | `{case.cost.shape[0]} x {case.cost.shape[1]}` |",
        f"| matched real pairs | `{len(result.assignment)}` |",
        f"| objective cost | `{result.objective_cost:.3f}` |",
        f"| real cost | `{result.real_cost:.3f}` |",
        f"| dual objective | `{result.dual_objective:.3f}` |",
        f"| trace frames | `{len(result.trace)}` |",
    ]

    if greedy is not None:
        summary_table.append(f"| greedy row-by-row cost | `{greedy['total']:.3f}` |")

    if brute is not None:
        summary_table.append(f"| brute-force cost | `{float(brute['objective_cost']):.3f}` |")
    return (
        brute,
        case,
        certificate,
        greedy,
        matched_lines,
        result,
        summary_table,
        unmatched,
    )


@app.cell
def _(case, matched_lines, mo, np, summary_table, unmatched):
    matched_md = "\n".join(matched_lines) if matched_lines else "- No real row was matched to a real column."
    unmatched_md = (
        "\n".join(f"- `{label}` matched to a dummy job" for label in unmatched)
        if unmatched
        else "- No dummy assignment was needed."
    )
    matrix_md = "```python\n" + np.array2string(case.cost, precision=2, suppress_small=True) + "\n```"
    selected_case_md = "\n".join(
        [
            "## Selected Case",
            "",
            f"**{case.name}**",
            "",
            case.description,
            "",
            "### Cost matrix",
            "",
            matrix_md,
            "",
            *summary_table,
            "",
            "### Optimal assignment",
            "",
            matched_md,
            "",
            "### Rows matched to dummy jobs",
            "",
            unmatched_md,
        ]
    )
    mo.md(selected_case_md)
    return


@app.cell
def _(
    case,
    plot_bipartite_assignment,
    plot_cost_heatmap,
    plot_spatial_assignment,
    plt,
    result,
):
    _figure, _axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_cost_heatmap(
        case.cost,
        assignment=result.assignment,
        row_labels=case.row_labels,
        col_labels=case.col_labels,
        ax=_axes[0],
        title="Original Cost Matrix",
        show_colorbar=True,
    )
    plot_cost_heatmap(
        result.reduced_cost[: case.cost.shape[0], : case.cost.shape[1]],
        assignment=result.assignment,
        row_labels=case.row_labels,
        col_labels=case.col_labels,
        ax=_axes[1],
        title="Final Reduced Costs",
        cmap="cividis",
        highlight_zeros=True,
        show_colorbar=True,
        annotation_fontsize=11,
    )

    if case.row_positions is not None and case.col_positions is not None:
        plot_spatial_assignment(
            case.row_positions,
            case.col_positions,
            result.assignment,
            row_labels=case.row_labels,
            col_labels=case.col_labels,
            ax=_axes[2],
            title="Spatial Matching View",
        )
    else:
        plot_bipartite_assignment(
            case.cost,
            result.assignment,
            row_labels=case.row_labels,
            col_labels=case.col_labels,
            ax=_axes[2],
            title="Bipartite Assignment Graph",
        )

    _figure.tight_layout()
    _figure
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## The Problem

    Given a cost matrix \(C \in \mathbb{R}^{n \times m}\), we want a one-to-one assignment with minimum total cost.

    Read the symbols as follows:

    - \(C = (c_{ij})\) is the full cost matrix.
    - \(c_{ij}\) means: the cost of assigning row \(i\) to column \(j\).
    - \(n\) is the number of rows, and \(m\) is the number of columns.
    - In the square case, \(n=m\), so every row can be matched with exactly one column and vice versa.

    For a square matrix, the combinatorial form is

    $$
    \min_{\pi \in S_n} \sum_{i=1}^{n} c_{i,\pi(i)}.
    $$

    Here is what each symbol means:

    - \(S_n\) is the set of all permutations of \(\{1,2,\dots,n\}\).
    - A **permutation** \(\pi\) is just a rule that reorders the column indices.
    - \(\pi(i)\) means: the column assigned to row \(i\).
    - So \(c_{i,\pi(i)}\) is the cost paid by row \(i\) under that assignment.
    - The sum \(\sum_{i=1}^{n} c_{i,\pi(i)}\) is the total cost of the full assignment.
    - The \(\min\) means: among all possible one-to-one assignments, choose the cheapest one.

    Example: if \(n=3\) and \(\pi=(2,1,3)\), then

    - row 1 is matched to column 2,
    - row 2 is matched to column 1,
    - row 3 is matched to column 3.

    That assignment costs

    $$
    c_{1,2} + c_{2,1} + c_{3,3}.
    $$

    In binary-variable form,

    $$
    \min \sum_{i,j} c_{ij} x_{ij}
    $$

    subject to

    $$
    \sum_j x_{ij} = 1 \quad \forall i,
    \qquad
    \sum_i x_{ij} = 1 \quad \forall j,
    \qquad
    x_{ij} \in \{0,1\}.
    $$

    In this version:

    - \(x_{ij}=1\) means row \(i\) is assigned to column \(j\).
    - \(x_{ij}=0\) means that assignment is not used.
    - \(\sum_j x_{ij}=1\) means each row chooses exactly one column.
    - \(\sum_i x_{ij}=1\) means each column is used by exactly one row.
    - \(x_{ij}\in\{0,1\}\) means the decision is yes/no, not fractional.

    For rectangular matrices, we pad with **dummy rows or dummy columns**. That turns the problem back into a square one.
    """).strip())
    return


@app.cell
def _(brute, case, dedent, greedy, math, mo, np, plt, result):
    _size = max(case.cost.shape)
    _ns = np.arange(1, min(9, _size + 4))
    _factorial_counts = [math.factorial(int(n)) for n in _ns]

    _figure, _axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _axes[0].plot(_ns, _factorial_counts, marker="o", color="#b5179e")
    _axes[0].set_yscale("log")
    _axes[0].set_title("Brute Force Search Space")
    _axes[0].set_xlabel("matrix size n")
    _axes[0].set_ylabel("number of permutations (log scale)")

    _labels = ["Hungarian"]
    _values = [result.objective_cost]
    _colors = ["#1d3557"]
    if greedy is not None:
        _labels.append("Greedy")
        _values.append(greedy["total"])
        _colors.append("#e76f51")
    if brute is not None:
        _labels.append("Brute force")
        _values.append(float(brute["objective_cost"]))
        _colors.append("#2a9d8f")

    _axes[1].bar(_labels, _values, color=_colors)
    _axes[1].set_title(f"Selected Case Cost Comparison ({case.name})")
    _axes[1].set_ylabel("objective value")

    _figure.tight_layout()

    _brute_note = (
        f"The exact brute-force optimum is `{float(brute['objective_cost']):.3f}`."
        if brute is not None
        else "Brute force was skipped because the padded matrix is larger than 8x8."
    )

    mo.vstack(
        [
            mo.md(
                dedent(
                    f"""
                **Why not brute force?**

                A square assignment of size `{_size}` already has `{math.factorial(_size):,}` possible permutations.
                {_brute_note}
                """
                ).strip()
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(
    case,
    initial_dual_from_reductions,
    np,
    plot_cost_heatmap,
    plt,
    row_reduction,
):
    _row_offsets, _row_reduced = row_reduction(case.cost)
    _, _col_offsets, _fully_reduced = initial_dual_from_reductions(case.cost)

    _figure, _axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_cost_heatmap(case.cost, row_labels=case.row_labels, col_labels=case.col_labels, ax=_axes[0], title="Original")
    plot_cost_heatmap(
        _row_reduced,
        row_labels=case.row_labels,
        col_labels=case.col_labels,
        ax=_axes[1],
        title=f"After Row Reduction\nrow minima = {np.round(_row_offsets, 2)}",
        highlight_zeros=True,
    )
    plot_cost_heatmap(
        _fully_reduced,
        row_labels=case.row_labels,
        col_labels=case.col_labels,
        ax=_axes[2],
        title=f"After Column Reduction\ncolumn minima = {np.round(_col_offsets, 2)}",
        highlight_zeros=True,
    )
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## Why Row and Column Reduction Is Valid

    Before the proof, define the symbols carefully:

    - \(c_{ij}\): the original cost in row \(i\), column \(j\).
    - \(a_i\): a number subtracted from every entry in row \(i\).
    - \(b_j\): a number subtracted from every entry in column \(j\).
    - \(c'_{ij}\): the new cost after those row and column shifts.
    - \(\pi\): one particular full assignment, viewed as a permutation.
    - \(\pi(i)\): the column chosen by row \(i\) in that assignment.

    So if row 2 is assigned to column 5, then \(\pi(2)=5\).

    Let \(a_i\) be any constants added to rows and \(b_j\) any constants added to columns. Define

    $$
    c'_{ij} = c_{ij} - a_i - b_j.
    $$

    This means:

    - first subtract \(a_i\) from the whole row \(i\),
    - then subtract \(b_j\) from the whole column \(j\).

    Now fix one assignment \(\pi\). Its total cost in the new matrix is

    For any perfect matching \(\pi\),

    $$
    \sum_i c'_{i,\pi(i)}
    =
    \sum_i c_{i,\pi(i)} - \sum_i a_i - \sum_i b_{\pi(i)}.
    $$

    Why is that true?

    - \(\sum_i c'_{i,\pi(i)}\) means: add the entries selected by the assignment in the new matrix.
    - Each selected entry changed from \(c_{i,\pi(i)}\) to \(c_{i,\pi(i)} - a_i - b_{\pi(i)}\).
    - Summing over all rows gives the formula above.

    Because \(\pi\) is a permutation, \(\sum_i b_{\pi(i)} = \sum_j b_j\). Therefore

    $$
    \sum_i c'_{i,\pi(i)}
    =
    \sum_i c_{i,\pi(i)} - \left(\sum_i a_i + \sum_j b_j\right).
    $$

    This is the key step.

    Since \(\pi\) is a permutation, it uses each column exactly once.
    So the list

    $$
    \pi(1), \pi(2), \dots, \pi(n)
    $$

    is just a reordering of

    $$
    1, 2, \dots, n.
    $$

    Therefore summing \(b_{\pi(i)}\) over all rows is the same as summing all column shifts once:

    $$
    \sum_i b_{\pi(i)} = \sum_j b_j.
    $$

    The term in parentheses is a **constant independent of the matching**. So every perfect matching is shifted by the same amount, and the minimizer does not change.

    In plain language:

    - every possible full assignment loses the same total amount,
    - so the ranking of assignments does not change,
    - therefore the assignment that was cheapest before is still cheapest after reduction.

    That single lemma justifies:

    - subtracting each row minimum,
    - subtracting each column minimum afterward,
    - repeatedly shifting uncovered and covered entries during the algorithm.
    """).strip())
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## The Dual View: Why Zeros Matter

    The LP dual of the assignment problem is

    $$
    \max \sum_i u_i + \sum_j v_j
    \qquad \text{subject to} \qquad
    u_i + v_j \le c_{ij}.
    $$

    Define the **reduced cost**

    $$
    r_{ij} = c_{ij} - u_i - v_j.
    $$

    Dual feasibility is exactly the statement

    $$
    r_{ij} \ge 0 \quad \forall i,j.
    $$

    Now take any perfect matching \(\pi\):

    $$
    \sum_i c_{i,\pi(i)}
    =
    \sum_i \left(u_i + v_{\pi(i)} + r_{i,\pi(i)}\right)
    =
    \sum_i u_i + \sum_j v_j + \sum_i r_{i,\pi(i)}.
    $$

    Since every reduced cost is nonnegative,

    $$
    \sum_i c_{i,\pi(i)} \ge \sum_i u_i + \sum_j v_j.
    $$

    So the dual objective is a lower bound on every assignment cost.

    If we find a perfect matching made only of zero reduced-cost edges, then

    $$
    \sum_i r_{i,\pi(i)} = 0,
    $$

    which forces

    $$
    \sum_i c_{i,\pi(i)} = \sum_i u_i + \sum_j v_j.
    $$

    At that moment, primal cost equals dual cost, so the matching is optimal.
    """).strip())
    return


@app.cell
def _(certificate, dedent, mo):
    mo.md(dedent(f"""
    ## Optimality Certificate for the Selected Case

    | certificate | value |
    | --- | --- |
    | dual feasible | `{certificate['dual_feasible']}` |
    | matched edges are tight | `{certificate['matched_edges_are_tight']}` |
    | primal / objective cost | `{certificate['objective_cost']:.6f}` |
    | dual objective | `{certificate['dual_objective']:.6f}` |
    | gap | `{certificate['gap']:.6e}` |

    The gap is numerically zero, which is exactly what strong duality predicts.
    """).strip())
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## Why the Dual Update Formula Works

    During the search, the algorithm maintains a set \(S\) of active rows and a set \(T\) of active columns.
    Let

    $$
    \delta = \min_{i \in S,\; j \notin T} r_{ij}.
    $$

    Then update

    $$
    u_i' =
    \begin{cases}
    u_i + \delta & i \in S \\
    u_i & i \notin S
    \end{cases}
    \qquad
    v_j' =
    \begin{cases}
    v_j - \delta & j \in T \\
    v_j & j \notin T.
    \end{cases}
    $$

    Check every type of edge:

    - If \(i \in S\) and \(j \notin T\), then \(r'_{ij} = r_{ij} - \delta \ge 0\) by definition of \(\delta\).
    - If \(i \notin S\) and \(j \in T\), then \(r'_{ij} = r_{ij} + \delta \ge 0\).
    - If both endpoints are in \(S \times T\) or both are outside, the reduced cost is unchanged.

    So dual feasibility is preserved.

    Also, at least one edge with \(i \in S\), \(j \notin T\) attains the minimum \(\delta\), so after the update

    $$
    r'_{ij} = 0.
    $$

    That creates at least one new zero edge, which expands the equality graph and lets the search continue.
    """).strip())
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## How to Read the Implementation

    The code in `algorithm.py` uses the classic competitive-programming form of the Hungarian algorithm.
    That implementation is fast, but the indexing is not obvious the first time you read it.

    The main convention is:

    - if the padded matrix is \(n \times n\), then several internal arrays have length \(n+1\),
    - index `0` is a **fake root column** used only during augmentation,
    - real rows and real columns live at indices `1..n` inside the implementation.

    So when the code writes

    ```python
    column_owner[0] = augmentation
    ```

    it means:

    - start a new augmenting-path search,
    - attach the new row to the fake root column,
    - then grow an alternating tree from that root.

    This notebook shows the public, zero-based version of those arrays in the trace explorer:

    - rows and columns are displayed as `0..n-1`,
    - unmatched or root references are shown as `-1`.
    """).strip())
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## Implementation Variable Map

    | code variable | meaning in the implementation |
    | --- | --- |
    | `row_potential` | the dual variables \(u\), stored with one extra slot because of the root convention |
    | `col_potential` | the dual variables \(v\), also stored with one extra slot |
    | `column_owner[j]` | which row currently owns column `j`; `0` means unmatched |
    | `way[j]` | predecessor column used to reconstruct the augmenting path |
    | `used_columns[j]` | whether column `j` is already inside the alternating tree |
    | `min_slack[j]` | the smallest reduced cost currently known from the tree to column `j` |
    | `current_column` | the column whose matched row is currently being expanded |
    | `next_column` | the uncovered column with smallest current slack |
    | `augmentation` | which row is currently being inserted into the matching |

    A good way to read the code is:

    1. initialize the dual variables with row and column reductions,
    2. insert one row at a time,
    3. grow an alternating tree over columns,
    4. maintain `min_slack` for all uncovered columns,
    5. shift the dual variables when no zero edge is immediately available,
    6. follow `way` backward to flip the matching.
    """).strip())
    return


@app.cell
def _(hungarian, initial_dual_from_reductions, inspect, mo):
    source_code = inspect.getsource(initial_dual_from_reductions) + "\n\n" + inspect.getsource(hungarian)
    mo.accordion({"Python implementation": mo.plain_text(source_code)})
    return


@app.cell
def _():
    stage_to_excerpt = {
        "Row and Column Reduction": """row_init, col_init, _ = initial_dual_from_reductions(padded)
    row_potential = np.zeros(size + 1, dtype=float)
    col_potential = np.zeros(size + 1, dtype=float)
    row_potential[1:] = row_init
    col_potential[1:] = col_init""",
        "Start Augmentation": """column_owner[0] = augmentation
    min_slack = np.full(size + 1, float("inf"))
    used_columns = np.zeros(size + 1, dtype=bool)
    way.fill(0)
    current_column = 0""",
        "Scan Uncovered Columns": """used_columns[current_column] = True
    row = column_owner[current_column]
    delta = float("inf")
    next_column = 0

    for col in range(1, size + 1):
    if used_columns[col]:
        continue
    slack = padded[row - 1, col - 1] - row_potential[row] - col_potential[col]
    if slack < min_slack[col] - EPS:
        min_slack[col] = slack
        way[col] = current_column
    if min_slack[col] < delta - EPS:
        delta = min_slack[col]
        next_column = col""",
        "Dual Update": """for col in range(size + 1):
    if used_columns[col]:
        row_potential[column_owner[col]] += delta
        col_potential[col] -= delta
    else:
        min_slack[col] -= delta

    current_column = next_column""",
        "Augment Matching": """while True:
    previous_column = way[current_column]
    column_owner[current_column] = column_owner[previous_column]
    current_column = previous_column
    if current_column == 0:
        break""",
    }
    return (stage_to_excerpt,)


@app.cell
def _(case, mo, result):
    trace_frame_slider = mo.ui.slider(
        start=0,
        stop=len(result.trace) - 1,
        step=1,
        value=0,
        include_input=True,
        show_value=True,
        label="Trace frame",
        full_width=True,
    )
    trace_row_labels_padded = case.row_labels + [
        f"dummy_worker_{index}" for index in range(len(case.row_labels), result.padded_cost.shape[0])
    ]
    trace_col_labels_padded = case.col_labels + [
        f"dummy_job_{index}" for index in range(len(case.col_labels), result.padded_cost.shape[1])
    ]
    return trace_col_labels_padded, trace_frame_slider, trace_row_labels_padded


@app.cell
def _(dedent, mo, result, trace_frame_slider):
    _frame_index = int(trace_frame_slider.value or 0)
    _frame = result.trace[_frame_index]
    _delta_text = f"{_frame.delta:.3f}" if _frame.delta is not None else "n/a"
    _trace_md = dedent(
        f"""
        ## Implementation Trace Explorer

        Use the controls to inspect the solver state frame by frame.

        - current frame: `{_frame_index}` of `{len(result.trace) - 1}`
        - augmentation: `{_frame.augmentation}`
        - stage: `{_frame.stage}`
        - root row being inserted: `{_frame.root_row if _frame.root_row is not None else 'n/a'}`
        - current column in the search: `{_frame.current_column if _frame.current_column is not None else 'ROOT'}`
        - next candidate column: `{_frame.next_column if _frame.next_column is not None else 'n/a'}`
        - matched edges in this frame: `{len(_frame.matching)}`
        - dual update delta: `{_delta_text}`

        Move the slider one frame at a time and compare the plot, the working arrays, and the matching update.
        """
    ).strip()
    mo.vstack(
        [
            mo.md(_trace_md),
            trace_frame_slider,
        ]
    )
    return


@app.cell
def _(
    inspect,
    mo,
    np,
    plot_trace_frame,
    plt,
    result,
    stage_to_excerpt,
    trace_col_labels_padded,
    trace_frame_slider,
    trace_row_labels_padded,
):
    _frame_index = int(trace_frame_slider.value or 0)
    _frame = result.trace[_frame_index]
    _figure, _axis = plt.subplots(figsize=(7.4, 5.8))
    plot_trace_frame(
        _frame,
        row_labels=trace_row_labels_padded,
        col_labels=trace_col_labels_padded,
        ax=_axis,
        cmap="cividis",
    )
    _potentials = np.array2string(_frame.row_potential, precision=2, suppress_small=True)
    _column_potentials = np.array2string(_frame.col_potential, precision=2, suppress_small=True)
    _owners = np.array2string(_frame.owner_by_column, separator=", ")
    _used_columns = np.array2string(_frame.used_columns.astype(int), separator=", ") if _frame.used_columns is not None else "n/a"
    _used_rows = np.array2string(_frame.used_rows.astype(int), separator=", ") if _frame.used_rows is not None else "n/a"
    _predecessor = (
        np.array2string(_frame.predecessor_column, separator=", ")
        if _frame.predecessor_column is not None
        else "n/a"
    )
    _min_slack = (
        np.array2string(_frame.min_slack_by_column, precision=2, suppress_small=True)
        if _frame.min_slack_by_column is not None
        else "n/a"
    )
    _code_excerpt = stage_to_excerpt.get(_frame.stage, inspect.cleandoc("No implementation excerpt registered for this stage."))
    mo.vstack(
        [
            _figure,
            mo.hstack(
                [
                    mo.md(
                        "\n".join(
                            [
                                "### Internal State at This Frame",
                                "",
                                f"- row potentials `u`: `{_potentials}`",
                                f"- column potentials `v`: `{_column_potentials}`",
                                f"- `column_owner[1:]`: `{_owners}`",
                                f"- `used_rows`: `{_used_rows}`",
                                f"- `used_columns`: `{_used_columns}`",
                                f"- `way[1:]` as predecessor columns: `{_predecessor}`",
                                f"- `min_slack[1:]`: `{_min_slack}`",
                                "",
                                _frame.note,
                            ]
                        )
                    ),
                    mo.vstack(
                        [
                            mo.md("### Code Block Running at This Stage"),
                            mo.plain_text(_code_excerpt),
                        ]
                    ),
                ],
                widths="equal",
                align="start",
                gap=1.0,
            ),
        ]
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## Step-by-Step Reading Strategy

            If you want to understand the implementation rather than memorize it, read one augmentation as:

            1. `Start Augmentation`: prepare fresh working arrays for one new row.
            2. `Scan Uncovered Columns`: compute slack values from the current tree to every uncovered column.
            3. `Dual Update`: if no free zero edge is available, shift the dual variables and reduce the slack.
            4. Repeat scanning and updating until an unmatched column is found.
            5. `Augment Matching`: follow `way` backward and flip ownership along the alternating path.

            The explorer above is built to mirror exactly that control flow.
            """
        ).strip()
    )
    return


@app.cell
def _(cases, dedent, hungarian, mo, plt):
    _suite_rows = []
    _case_names = list(cases.keys())
    _suite_costs = []

    for name, _case in cases.items():
        _solved = hungarian(_case.cost, record_trace=False)
        _suite_costs.append(_solved.objective_cost)
        _suite_rows.append(
            f"| {name} | `{_case.cost.shape[0]} x {_case.cost.shape[1]}` | `{_solved.objective_cost:.3f}` | `{len(_solved.assignment)}` |"
        )

    _figure, _axis = plt.subplots(figsize=(10, 4.5))
    _axis.bar(_case_names, _suite_costs, color="#457b9d")
    _axis.set_ylabel("objective cost")
    _axis.set_title("Optimal Cost Across Synthetic Cases")
    _axis.tick_params(axis="x", rotation=20)
    _figure.tight_layout()

    mo.vstack(
        [
            mo.md(
                dedent(
                    """
                ## Synthetic Case Suite

                | case | shape | objective cost | real matches |
                | --- | --- | --- | --- |
                """
                ).strip()
                + "\n"
                + "\n".join(_suite_rows)
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(brute_force_assignment, dedent, hungarian, mo, np, plt, time):
    _rng = np.random.default_rng(11)

    _validation_trials = 30
    _passed = 0
    for _ in range(_validation_trials):
        _n_rows = int(_rng.integers(2, 6))
        _n_cols = int(_rng.integers(2, 6))
        _cost = _rng.integers(0, 25, size=(_n_rows, _n_cols)).astype(float)
        _hungarian_result = hungarian(_cost, record_trace=False)
        _brute_result = brute_force_assignment(_cost)
        if np.isclose(_hungarian_result.objective_cost, _brute_result["objective_cost"]):
            _passed += 1

    _sizes = np.array([2, 4, 6, 8, 10, 12, 16, 20, 24, 30])
    _runtimes_ms = []
    for _size in _sizes:
        _samples = []
        for seed in range(3):
            _matrix = np.random.default_rng(seed).integers(0, 100, size=(int(_size), int(_size))).astype(float)
            _start = time.perf_counter()
            hungarian(_matrix, record_trace=False)
            _samples.append((time.perf_counter() - _start) * 1_000.0)
        _runtimes_ms.append(float(np.mean(_samples)))

    _figure, _axis = plt.subplots(figsize=(8, 4.5))
    _axis.plot(_sizes, _runtimes_ms, marker="o", color="#6a4c93")
    _axis.set_title("Empirical Runtime of the Hungarian Implementation")
    _axis.set_xlabel("matrix size n")
    _axis.set_ylabel("average runtime (ms)")
    _figure.tight_layout()

    mo.vstack(
        [
            mo.md(
                dedent(
                    f"""
                ## Validation and Scaling

                On `{_validation_trials}` random small matrices, the Hungarian implementation matched brute force on
                `{_passed}/{_validation_trials}` cases.

                The runtime curve below is empirical. The expected complexity is cubic in the square case, roughly $O(n^3)$.
                """
                ).strip()
            ),
            _figure,
        ]
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(dedent(r"""
    ## Final Takeaways

    1. The Hungarian algorithm is not magic: it is a **primal-dual method** for the assignment problem.
    2. Row and column reductions are valid because they shift every perfect matching by the same constant.
    3. Zeros matter because they are edges with **zero reduced cost**, where primal and dual can meet.
    4. When a perfect matching exists in the equality graph, optimality is certified immediately.
    5. The dual update preserves feasibility and creates new zero edges, which is the engine of progress.

    If you want to keep exploring, the easiest next step is to switch between the synthetic cases at the top and compare:

    - the original heatmap,
    - the reduced-cost heatmap,
    - the graph or spatial view,
    - the search animation.
    """).strip())
    return


if __name__ == "__main__":
    app.run()
