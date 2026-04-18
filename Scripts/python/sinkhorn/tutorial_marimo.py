import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import inspect
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

    from algorithm import gibbs_kernel, sinkhorn
    from synthetic_cases import default_case_suite
    from visualization import (
        plot_convergence,
        plot_distribution,
        plot_matrix_heatmap,
        plot_trace_frame,
        plot_transport_graph,
    )

    return (
        dedent,
        default_case_suite,
        gibbs_kernel,
        inspect,
        mo,
        np,
        plot_convergence,
        plot_distribution,
        plot_matrix_heatmap,
        plot_trace_frame,
        plot_transport_graph,
        plt,
        sinkhorn,
        time,
    )


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            # Sinkhorn Algorithm, Really Explained

            This notebook is a full tutorial on the Sinkhorn algorithm for **entropic optimal transport**.

            It is built with the same philosophy as the Hungarian notebook:

            - understand the optimization problem,
            - derive the update rules from the math,
            - read the Python implementation variable by variable,
            - inspect every iteration with controls and plots.

            The implementation here is the transparent scaling form of Sinkhorn:

            $$
            u \leftarrow \frac{a}{K v},
            \qquad
            v \leftarrow \frac{b}{K^\top u},
            \qquad
            K = e^{-C / \varepsilon}.
            $$
            """
        ).strip()
    )
    return


@app.cell
def _(default_case_suite, mo):
    case_selector = mo.ui.dropdown(
        options=list(default_case_suite(seed=7).keys()),
        value="Classic 1D Shift",
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
    epsilon_slider = mo.ui.slider(
        start=0.05,
        stop=1.50,
        step=0.05,
        value=0.35,
        include_input=True,
        show_value=True,
        label="Entropic regularization ε",
        full_width=True,
    )
    max_iterations_slider = mo.ui.slider(
        start=20,
        stop=300,
        step=10,
        value=120,
        include_input=True,
        show_value=True,
        label="Max iterations",
        full_width=True,
    )
    return case_selector, epsilon_slider, max_iterations_slider, seed_slider


@app.cell
def _(case_selector, epsilon_slider, max_iterations_slider, mo, seed_slider):
    mo.vstack([case_selector, seed_slider, epsilon_slider, max_iterations_slider])
    return


@app.cell
def _(default_case_suite, seed_slider):
    cases = default_case_suite(seed=seed_slider.value)
    return (cases,)


@app.cell
def _(case_selector, cases, epsilon_slider, max_iterations_slider, sinkhorn):
    case = cases[case_selector.value]
    epsilon = float(epsilon_slider.value)
    max_iterations = int(max_iterations_slider.value)
    result = sinkhorn(
        case.cost_matrix,
        case.source_weights,
        case.target_weights,
        epsilon=epsilon,
        max_iterations=max_iterations,
        tol=1e-10,
        record_trace=True,
    )
    return case, epsilon, max_iterations, result


@app.cell
def _(case, epsilon, mo, np, result):
    source_md = "```python\n" + np.array2string(case.source_weights, precision=3, suppress_small=True) + "\n```"
    target_md = "```python\n" + np.array2string(case.target_weights, precision=3, suppress_small=True) + "\n```"
    cost_md = "```python\n" + np.array2string(case.cost_matrix, precision=3, suppress_small=True) + "\n```"
    summary_md = "\n".join(
        [
            "## Selected Sinkhorn Case",
            "",
            f"**{case.name}**",
            "",
            case.description,
            "",
            "| quantity | value |",
            "| --- | --- |",
            f"| source size | `{case.source_weights.shape[0]}` |",
            f"| target size | `{case.target_weights.shape[0]}` |",
            f"| epsilon | `{epsilon:.3f}` |",
            f"| converged | `{result.converged}` |",
            f"| iterations run | `{result.iterations_run}` |",
            f"| final row L1 error | `{result.row_l1_history[-1] if result.row_l1_history.size else 0.0:.3e}` |",
            f"| final col L1 error | `{result.col_l1_history[-1] if result.col_l1_history.size else 0.0:.3e}` |",
            f"| transport cost | `{result.transport_cost:.6f}` |",
            f"| regularized objective | `{result.regularized_objective:.6f}` |",
            "",
            "### Source weights a",
            "",
            source_md,
            "",
            "### Target weights b",
            "",
            target_md,
            "",
            "### Cost matrix C",
            "",
            cost_md,
        ]
    )
    mo.md(summary_md)
    return


@app.cell
def _(case, plot_distribution, plot_transport_graph, plt, result):
    _figure = plt.figure(figsize=(14, 4.8))
    _axes = _figure.subplots(1, 3)

    plot_distribution(
        case.source_weights,
        labels=case.source_labels,
        ax=_axes[0],
        title="Source distribution a",
        color="#457b9d",
    )
    plot_distribution(
        case.target_weights,
        labels=case.target_labels,
        ax=_axes[1],
        title="Target distribution b",
        color="#e76f51",
    )
    plot_transport_graph(
        case.source_positions,
        case.target_positions,
        case.source_weights,
        case.target_weights,
        result.coupling,
        source_labels=case.source_labels,
        target_labels=case.target_labels,
        ax=_axes[2],
        title="Final transport plan",
    )
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(case, plot_matrix_heatmap, plt, result):
    _figure = plt.figure(figsize=(16, 4.8))
    _axes = _figure.subplots(1, 3)

    plot_matrix_heatmap(
        case.cost_matrix,
        row_labels=case.source_labels,
        col_labels=case.target_labels,
        ax=_axes[0],
        title="Cost matrix C",
        cmap="YlGnBu",
        show_colorbar=True,
        value_fmt="{:.2f}",
    )
    plot_matrix_heatmap(
        result.kernel,
        row_labels=case.source_labels,
        col_labels=case.target_labels,
        ax=_axes[1],
        title="Gibbs kernel K = exp(-C / ε)",
        cmap="viridis",
        show_colorbar=True,
        value_fmt="{:.3f}",
    )
    plot_matrix_heatmap(
        result.coupling,
        row_labels=case.source_labels,
        col_labels=case.target_labels,
        ax=_axes[2],
        title="Final coupling P",
        cmap="magma",
        show_colorbar=True,
        value_fmt="{:.3f}",
    )
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            ## The Optimal Transport Problem

            We want to transport a source probability vector \(a \in \mathbb{R}^n\) onto a target probability vector \(b \in \mathbb{R}^m\).

            Definitions:

            - \(a_i\): mass available at source location \(i\),
            - \(b_j\): mass demanded at target location \(j\),
            - \(C_{ij}\): cost of sending one unit of mass from source \(i\) to target \(j\),
            - \(P_{ij}\): how much mass we actually send from \(i\) to \(j\).

            The transport plan \(P\) must satisfy:

            $$
            P \mathbf{1} = a,
            \qquad
            P^\top \mathbf{1} = b,
            \qquad
            P_{ij} \ge 0.
            $$

            Read those constraints as:

            - each row sum of \(P\) must match the source mass \(a_i\),
            - each column sum of \(P\) must match the target mass \(b_j\),
            - no transported mass can be negative.

            The classical optimal transport objective is

            $$
            \min_{P \ge 0} \langle C, P \rangle
            \quad \text{subject to the marginal constraints above.}
            $$

            Here \(\langle C, P \rangle = \sum_{i,j} C_{ij} P_{ij}\) is the total transport cost.
            """
        ).strip()
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            ## Entropic Regularization

            Sinkhorn solves the **entropically regularized** problem

            $$
            \min_{P \ge 0}
            \sum_{i,j} C_{ij} P_{ij}
            +
            \varepsilon \sum_{i,j} P_{ij}(\log P_{ij} - 1)
            $$

            subject to

            $$
            P \mathbf{1} = a,
            \qquad
            P^\top \mathbf{1} = b.
            $$

            What changes when we add the entropy term?

            - small \(\varepsilon\): the plan becomes sharper and closer to sparse optimal transport,
            - large \(\varepsilon\): the plan becomes smoother and more diffuse,
            - computationally: the optimization becomes much easier because the solution factorizes nicely.
            """
        ).strip()
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            ## Why the Solution Has the Form P = diag(u) K diag(v)

            Introduce Lagrange multipliers \(f \in \mathbb{R}^n\) and \(g \in \mathbb{R}^m\) for the row and column constraints.
            If you differentiate the regularized objective with respect to \(P_{ij}\), the first-order condition gives

            $$
            C_{ij} + \varepsilon \log P_{ij} - f_i - g_j = 0.
            $$

            Rearranging,

            $$
            \log P_{ij} = \frac{f_i + g_j - C_{ij}}{\varepsilon}.
            $$

            Exponentiating both sides,

            $$
            P_{ij} = \exp(f_i / \varepsilon) \exp(-C_{ij}/\varepsilon) \exp(g_j/\varepsilon).
            $$

            Now define

            $$
            u_i = \exp(f_i/\varepsilon),
            \qquad
            v_j = \exp(g_j/\varepsilon),
            \qquad
            K_{ij} = \exp(-C_{ij}/\varepsilon).
            $$

            Then the solution must have the factorized form

            $$
            P_{ij} = u_i K_{ij} v_j,
            \qquad
            P = \operatorname{diag}(u) K \operatorname{diag}(v).
            $$

            That factorization is the entire reason Sinkhorn becomes a scaling algorithm.
            """
        ).strip()
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            ## Why the Sinkhorn Updates Work

            Suppose \(v\) is fixed. Then

            $$
            P = \operatorname{diag}(u) K \operatorname{diag}(v)
            $$

            has row sums

            $$
            (P \mathbf{1})_i = u_i \sum_j K_{ij} v_j = u_i (K v)_i.
            $$

            To enforce the row constraint \(P \mathbf{1} = a\), we must choose

            $$
            u_i = \frac{a_i}{(K v)_i}.
            $$

            Vector form:

            $$
            u = \frac{a}{K v}.
            $$

            By the same logic, if \(u\) is fixed, the column constraint requires

            $$
            v = \frac{b}{K^\top u}.
            $$

            So Sinkhorn alternates:

            1. scale rows exactly,
            2. scale columns exactly,
            3. repeat until both sets of constraints are simultaneously almost satisfied.
            """
        ).strip()
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            ## Reading the Implementation

            The implementation in `algorithm.py` is deliberately explicit:

            - `kernel` is \(K = e^{-C / \varepsilon}\),
            - `u` is the current row scaling vector,
            - `v` is the current column scaling vector,
            - `coupling = u[:, None] * kernel * v[None, :]`,
            - `row_sum` and `col_sum` are the current marginals of the coupling,
            - `row_error` and `col_error` measure how far those marginals are from \(a\) and \(b\).

            The notebook trace records the state after initialization, after each `u` update, after each `v` update, and after final assembly.

            This is the simplest way to understand Sinkhorn from code:

            1. compute `kernel`,
            2. update `u`,
            3. inspect the row sums,
            4. update `v`,
            5. inspect the column sums,
            6. repeat until the errors are tiny.
            """
        ).strip()
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            ## Implementation Variable Map

            | variable | meaning |
            | --- | --- |
            | `cost_matrix` | the transport cost matrix \(C\) |
            | `epsilon` | the entropic regularization strength \(\varepsilon\) |
            | `kernel` | the Gibbs kernel \(K = \exp(-C / \varepsilon)\) |
            | `u` | row scaling vector |
            | `v` | column scaling vector |
            | `coupling` | current transport plan \(P = \mathrm{diag}(u)K\mathrm{diag}(v)\) |
            | `row_sum` | current row marginals of \(P\) |
            | `col_sum` | current column marginals of \(P\) |
            | `row_error` | `row_sum - a` |
            | `col_error` | `col_sum - b` |
            | `dual_f` | \(\varepsilon \log u\) |
            | `dual_g` | \(\varepsilon \log v\) |
            """
        ).strip()
    )
    return


@app.cell
def _(gibbs_kernel, inspect, mo, sinkhorn):
    source_code = inspect.getsource(gibbs_kernel) + "\n\n" + inspect.getsource(sinkhorn)
    mo.accordion({"Python implementation": mo.plain_text(source_code)})
    return


@app.cell
def _():
    stage_to_excerpt = {
        "Initialize Kernel": """kernel = np.exp(-cost / epsilon)
u = np.ones_like(source)
v = np.ones_like(target)""",
        "Update u": """u = source / np.maximum(kernel @ v, EPS)""",
        "Update v": """v = target / np.maximum(kernel.T @ u, EPS)""",
        "Final Coupling": """coupling = u[:, None] * kernel * v[None, :]""",
    }
    return (stage_to_excerpt,)


@app.cell
def _(mo, result):
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
    return (trace_frame_slider,)


@app.cell
def _(dedent, mo, result, trace_frame_slider):
    _frame_index = int(trace_frame_slider.value or 0)
    _frame = result.trace[_frame_index]
    _trace_md = dedent(
        f"""
        ## Sinkhorn Trace Explorer

        Move through the iteration history one frame at a time.

        - current frame: `{_frame_index}` of `{len(result.trace) - 1}`
        - iteration: `{_frame.iteration}`
        - stage: `{_frame.stage}`
        - row L1 error: `{_frame.row_l1_error:.3e}`
        - column L1 error: `{_frame.col_l1_error:.3e}`
        - transport cost: `{_frame.transport_cost:.6f}`
        - regularized objective: `{_frame.regularized_objective:.6f}`
        """
    ).strip()
    mo.vstack([mo.md(_trace_md), trace_frame_slider])
    return


@app.cell
def _(case, mo, np, plot_trace_frame, plt, result, stage_to_excerpt, trace_frame_slider):
    _frame_index = int(trace_frame_slider.value or 0)
    _frame = result.trace[_frame_index]
    _figure = plot_trace_frame(
        _frame,
        case.source_weights,
        case.target_weights,
        source_labels=case.source_labels,
        target_labels=case.target_labels,
    )
    _u_text = np.array2string(_frame.u, precision=4, suppress_small=True)
    _v_text = np.array2string(_frame.v, precision=4, suppress_small=True)
    _f_text = np.array2string(_frame.dual_f, precision=4, suppress_small=True)
    _g_text = np.array2string(_frame.dual_g, precision=4, suppress_small=True)
    _code_excerpt = stage_to_excerpt.get(_frame.stage, "No code excerpt registered for this stage.")
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
                                f"- scaling vector `u`: `{_u_text}`",
                                f"- scaling vector `v`: `{_v_text}`",
                                f"- dual potential `f = ε log u`: `{_f_text}`",
                                f"- dual potential `g = ε log v`: `{_g_text}`",
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
    plt.close(_figure)
    return


@app.cell
def _(plot_convergence, plt, result):
    _figure, _axis = plt.subplots(figsize=(8, 4.5))
    plot_convergence(result, ax=_axis, title="Marginal error decay")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(case, epsilon, max_iterations, plot_matrix_heatmap, plt, sinkhorn):
    _epsilon_small = max(0.05, epsilon / 2.0)
    _epsilon_large = min(1.50, max(epsilon + 0.25, epsilon * 2.0))
    if abs(_epsilon_large - _epsilon_small) < 1e-9:
        _epsilon_large = min(1.50, _epsilon_small + 0.25)

    _sharp = sinkhorn(
        case.cost_matrix,
        case.source_weights,
        case.target_weights,
        epsilon=_epsilon_small,
        max_iterations=max_iterations,
        tol=1e-10,
        record_trace=False,
    )
    _smooth = sinkhorn(
        case.cost_matrix,
        case.source_weights,
        case.target_weights,
        epsilon=_epsilon_large,
        max_iterations=max_iterations,
        tol=1e-10,
        record_trace=False,
    )

    _figure = plt.figure(figsize=(12.5, 4.8))
    _axes = _figure.subplots(1, 2)
    plot_matrix_heatmap(
        _sharp.coupling,
        row_labels=case.source_labels,
        col_labels=case.target_labels,
        ax=_axes[0],
        title=f"Smaller ε = {_epsilon_small:.2f}",
        cmap="magma",
        show_colorbar=True,
        value_fmt="{:.3f}",
    )
    plot_matrix_heatmap(
        _smooth.coupling,
        row_labels=case.source_labels,
        col_labels=case.target_labels,
        ax=_axes[1],
        title=f"Larger ε = {_epsilon_large:.2f}",
        cmap="magma",
        show_colorbar=True,
        value_fmt="{:.3f}",
    )
    _figure.tight_layout()
    _figure
    epsilon_large_comparison = _epsilon_large
    epsilon_small_comparison = _epsilon_small
    sharp_result_comparison = _sharp
    smooth_result_comparison = _smooth
    return (
        epsilon_large_comparison,
        epsilon_small_comparison,
        sharp_result_comparison,
        smooth_result_comparison,
    )


@app.cell
def _(dedent, epsilon_large_comparison, epsilon_small_comparison, mo, sharp_result_comparison, smooth_result_comparison):
    mo.md(
        dedent(
            f"""
            ## Effect of the Regularization Parameter ε

            Compare the two couplings above:

            - with smaller `ε = {epsilon_small_comparison:.2f}`, the plan is sharper and closer to a sparse transport map,
            - with larger `ε = {epsilon_large_comparison:.2f}`, the plan is more spread out because entropy is weighted more heavily.

            Final objectives:

            - smaller ε regularized objective: `{sharp_result_comparison.regularized_objective:.6f}`
            - larger ε regularized objective: `{smooth_result_comparison.regularized_objective:.6f}`
            """
        ).strip()
    )
    return


@app.cell
def _(cases, dedent, epsilon, max_iterations, mo, sinkhorn):
    _rows = []
    for _name, _case in cases.items():
        _result = sinkhorn(
            _case.cost_matrix,
            _case.source_weights,
            _case.target_weights,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tol=1e-10,
            record_trace=False,
        )
        _final_row_error = _result.row_l1_history[-1] if _result.row_l1_history.size else 0.0
        _final_col_error = _result.col_l1_history[-1] if _result.col_l1_history.size else 0.0
        _rows.append(
            f"| {_name} | `{_result.iterations_run}` | `{_final_row_error:.2e}` | `{_final_col_error:.2e}` | `{_result.transport_cost:.4f}` |"
        )

    mo.md(
        dedent(
            """
            ## Synthetic Case Suite

            | case | iterations | final row error | final col error | transport cost |
            | --- | --- | --- | --- | --- |
            """
        ).strip()
        + "\n"
        + "\n".join(_rows)
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## Final Takeaways

            1. Entropic regularization turns optimal transport into a matrix scaling problem.
            2. The Gibbs kernel `K = exp(-C / ε)` is fixed for a chosen `ε`.
            3. Sinkhorn alternates exact row scaling and exact column scaling.
            4. The transport plan is always reconstructed as `P = diag(u) K diag(v)`.
            5. Small `ε` gives sharper plans; large `ε` gives smoother plans.
            6. The implementation is easiest to understand by following `u`, `v`, the current coupling, and the marginal errors side by side.
            """
        ).strip()
    )
    return


if __name__ == "__main__":
    app.run()
