"""Build publication figures and LaTeX tables from the final JCP experiments.

The script deliberately uses only the immutable final CSV snapshots tracked in
``reproducibility/results``. Test 2 PINN rows come from the standard combined
snapshot and its classical rows are replaced by the stronger expanding-window
time-CV snapshot.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARTICLE = Path(__file__).resolve().parent
PAPER = ARTICLE.parent
ROOT = PAPER.parent
RESULTS = ROOT / "reproducibility" / "results"
OUT = ARTICLE / "generated"
FIGURES = ARTICLE / "figures"
METHODS = ["H-PINN", "HO-PINN", "XGB", "SVM", "GP"]
COLORS = {
    "H-PINN": "#c43c39",
    "HO-PINN": "#ef8a62",
    "XGB": "#37659e",
    "SVM": "#3a923a",
    "GP": "#7956b2",
}
MARKERS = {"H-PINN": "o", "HO-PINN": "s", "XGB": "^", "SVM": "D", "GP": "v"}


def load_results() -> pd.DataFrame:
    sources = {
        "Test 1": RESULTS / "test1.csv",
        "Test 2": RESULTS / "test2_pinn_and_standard_classical.csv",
        "Test 3": RESULTS / "test3.csv",
        "Test 4": RESULTS / "test4.csv",
    }
    frames = []
    for test, path in sources.items():
        frame = pd.read_csv(path)
        frame["test"] = test
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    strong = pd.read_csv(
        RESULTS / "test2_strong_classical.csv"
    )
    strong["test"] = "Test 2"
    is_test2_classical = (data["test"] == "Test 2") & data["method"].isin(
        ["XGB", "SVM", "GP"]
    )
    data = pd.concat([data.loc[~is_test2_classical], strong], ignore_index=True)

    key = ["test", "evaluation_region", "n_train", "seed", "method"]
    if data.duplicated(key).any():
        raise RuntimeError("Duplicate final result rows detected")
    expected = 4 * 2 * 6 * 3 * 5
    if len(data) != expected:
        raise RuntimeError(f"Expected {expected} final rows, found {len(data)}")
    return data


def metric_for(test: str, suffix: str) -> str:
    return suffix if test == "Test 1" else f"Swat_{suffix}"


def fmt(mean: float, std: float, *, digits: int = 3, bold: bool = False) -> str:
    value = f"{mean:.{digits}f}\\,$\\pm$\\,{std:.{digits}f}"
    return f"\\textbf{{{value}}}" if bold else value


def fmt_sci(mean: float, std: float, *, bold: bool = False) -> str:
    value = f"$\\num{{{mean:.2e}}}\\,\\pm\\,\\num{{{std:.2e}}}$"
    return f"\\textbf{{{value}}}" if bold else value


def summary(data: pd.DataFrame, test: str, region: str, budget: int, metric: str):
    subset = data[
        (data.test == test)
        & (data.evaluation_region == region)
        & (data.n_train == budget)
    ]
    return subset.groupby("method")[metric].agg(["mean", "std"])


def write_primary_table(data: pd.DataFrame) -> None:
    lines = [
        "\\begin{sidewaystable}[!htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Primary held-out $R^2$ at two scarce-label budgets (mean $\\pm$ sample standard deviation over seeds 101, 102, and 103). Test 1 uses the scalar temperature; Tests 2--4 use water saturation $S_w$. Bold marks the highest mean within a row.}",
        "\\label{tab:primary_r2}",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Test & Region & $N$ & H-PINN & HO-PINN & XGB & SVM & GP " + "\\\\",
        "\\midrule",
    ]
    for test in [f"Test {i}" for i in range(1, 5)]:
        metric = metric_for(test, "R2")
        for region in ["interpolation", "extrapolation"]:
            for budget in [20, 100]:
                stats = summary(data, test, region, budget, metric)
                best = stats["mean"].max()
                values = [
                    fmt(stats.loc[m, "mean"], stats.loc[m, "std"], bold=stats.loc[m, "mean"] == best)
                    for m in METHODS
                ]
                label = "Interp." if region == "interpolation" else "Extrap."
                lines.append(f"{test} & {label} & {budget} & " + " & ".join(values) + " " + "\\\\")
        if test != "Test 4":
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{sidewaystable}", ""])
    (OUT / "primary_r2_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_flow_target_table(data: pd.DataFrame, target: str) -> None:
    target_name = "pressure" if target == "Pressure" else "water saturation"
    stem = target.lower()
    lines = [
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{longtable}{lllcc}",
        f"\\caption{{Held-out {target_name} $R^2$ and MSE at $N=50$ (mean $\\pm$ sample standard deviation over seeds 101--103). Bold marks the highest $R^2$ and lowest MSE within each test--region block.}}\\label{{tab:{stem}_r2_mse_n50}}" + "\\\\",
        "\\toprule",
        "Test & Region & Method & $R^2$ & MSE " + "\\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{5}{c}{\\tablename~\\thetable{} continued} " + "\\\\",
        "\\toprule",
        "Test & Region & Method & $R^2$ & MSE " + "\\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{5}{r}{Continued on next page} " + "\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for test in ["Test 2", "Test 3", "Test 4"]:
        for region in ["interpolation", "extrapolation"]:
            r2 = summary(data, test, region, 50, f"{target}_R2")
            mse = summary(data, test, region, 50, f"{target}_MSE")
            best_r2 = r2["mean"].max()
            best_mse = mse["mean"].min()
            for method in METHODS:
                region_label = "Interp." if region == "interpolation" else "Extrap."
                lines.append(
                    f"{test} & {region_label} & {method} & "
                    f"{fmt(r2.loc[method, 'mean'], r2.loc[method, 'std'], bold=np.isclose(r2.loc[method, 'mean'], best_r2))} & "
                    f"{fmt_sci(mse.loc[method, 'mean'], mse.loc[method, 'std'], bold=np.isclose(mse.loc[method, 'mean'], best_mse))} "
                    + "\\\\"
                )
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\end{longtable}", "\\endgroup", ""])
    (OUT / f"{stem}_r2_mse_n50_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_flow_tables(data: pd.DataFrame) -> None:
    write_flow_target_table(data, "Swat")
    write_flow_target_table(data, "Pressure")


def write_runtime_table(data: pd.DataFrame) -> None:
    unique = data[data.n_train == 50].drop_duplicates(["test", "seed", "method"])
    stats = unique.groupby(["test", "method"])["train_time_sec"].agg(["mean", "std"])
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Wall-clock fitting and tuning time in seconds at $N=50$ (mean $\\pm$ sample standard deviation over three seeds). Classical times include hyperparameter selection; reference-data generation is excluded.}",
        "\\label{tab:runtime}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Test & H-PINN & HO-PINN & XGB & SVM & GP " + "\\\\",
        "\\midrule",
    ]
    for test in [f"Test {i}" for i in range(1, 5)]:
        best = min(stats.loc[test, "mean"])
        values = [
            fmt(stats.loc[(test, m), "mean"], stats.loc[(test, m), "std"], digits=2, bold=stats.loc[(test, m), "mean"] == best)
            for m in METHODS
        ]
        lines.append(f"{test} & " + " & ".join(values) + " " + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (OUT / "runtime_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_secondary_metric_table(data: pd.DataFrame, target: str) -> None:
    stem = target.lower()
    target_name = "Pressure" if target == "Pressure" else "water saturation"
    metrics = ["RMSE", "MAE", "relative_L2", "NRMSE_std"]
    lines = [
        "\\begingroup",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.25pt}",
        "\\begin{longtable}{lllcccc}",
        f"\\caption{{Secondary {target_name} metrics at $N=50$ (mean $\\pm$ sample standard deviation over seeds 101--103).}}\\label{{tab:appendix_{stem}_metrics}}" + "\\\\",
        "\\toprule",
        "Test & Region & Method & RMSE & MAE & relative $L^2$ & NRMSE$_{\\mathrm{std}}$ " + "\\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{7}{c}{\\tablename~\\thetable{} continued} " + "\\\\",
        "\\toprule",
        "Test & Region & Method & RMSE & MAE & relative $L^2$ & NRMSE$_{\\mathrm{std}}$ " + "\\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{7}{r}{Continued on next page} " + "\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for test in ["Test 2", "Test 3", "Test 4"]:
        for region in ["interpolation", "extrapolation"]:
            for method in METHODS:
                values = []
                for metric in metrics:
                    stats = summary(data, test, region, 50, f"{target}_{metric}")
                    values.append(fmt(stats.loc[method, "mean"], stats.loc[method, "std"], digits=4))
                region_label = "Interp." if region == "interpolation" else "Extrap."
                lines.append(f"{test} & {region_label} & {method} & " + " & ".join(values) + " " + "\\\\")
            lines.append("\\addlinespace[2pt]")
    lines.extend(["\\end{longtable}", "\\endgroup", ""])
    (OUT / f"appendix_{stem}_metrics_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_physical_diagnostics_table(data: pd.DataFrame) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Physical diagnostics on the extrapolation region at $N=50$ (mean $\\pm$ sample standard deviation). The violation rate is the fraction of predictions outside the tolerated saturation interval; closure is $|\\widehat S_w+\\widehat S_o-1|$.}",
        "\\label{tab:physical_diagnostics}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Test & Method & Bounds violation & Closure MAE " + "\\\\",
        "\\midrule",
    ]
    for test in ["Test 2", "Test 3", "Test 4"]:
        for method in METHODS:
            violation = summary(data, test, "extrapolation", 50, "saturation_bounds_violation").loc[method]
            closure = summary(data, test, "extrapolation", 50, "closure_MAE").loc[method]
            lines.append(
                f"{test} & {method} & {fmt(violation['mean'], violation['std'], digits=4)} & "
                f"{fmt_sci(closure['mean'], closure['std'])} " + "\\\\"
            )
        if test != "Test 4":
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (OUT / "physical_diagnostics_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_test1_secondary_table(data: pd.DataFrame) -> None:
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\caption{Complete Test 1 metrics at $N=50$ (mean $\\pm$ sample standard deviation over seeds 101--103).}",
        "\\label{tab:test1_secondary}",
        "\\textbf{(a) Absolute errors}\\par\\smallskip",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Region & Method & MSE & RMSE & MAE " + "\\\\",
        "\\midrule",
    ]
    for region in ["interpolation", "extrapolation"]:
        for method in METHODS:
            values = []
            for metric in ["MSE", "RMSE", "MAE"]:
                stats = summary(data, "Test 1", region, 50, metric)
                if metric == "MSE":
                    values.append(fmt_sci(stats.loc[method, "mean"], stats.loc[method, "std"]))
                else:
                    values.append(fmt(stats.loc[method, "mean"], stats.loc[method, "std"], digits=4))
            region_label = "Interp." if region == "interpolation" else "Extrap."
            lines.append(f"{region_label} & {method} & " + " & ".join(values) + " " + "\\\\")
        if region == "interpolation":
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule", "\\end{tabular}", "\\par\\medskip",
        "\\textbf{(b) Coefficient of determination and normalized errors}\\par\\smallskip",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Region & Method & $R^2$ & relative $L^2$ & NRMSE$_{\\mathrm{std}}$ " + "\\\\",
        "\\midrule",
    ])
    for region in ["interpolation", "extrapolation"]:
        for method in METHODS:
            values = []
            for metric in ["R2", "relative_L2", "NRMSE_std"]:
                stats = summary(data, "Test 1", region, 50, metric)
                values.append(fmt(stats.loc[method, "mean"], stats.loc[method, "std"], digits=4))
            region_label = "Interp." if region == "interpolation" else "Extrap."
            lines.append(f"{region_label} & {method} & " + " & ".join(values) + " " + "\\\\")
        if region == "interpolation":
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (OUT / "test1_secondary_metrics_table.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_scaling(data: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    budgets = np.array([20, 50, 100, 200, 500, 1000])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), sharex=True)
    for ax, region in zip(axes, ["interpolation", "extrapolation"]):
        for method in METHODS:
            sub = data[(data.test == "Test 1") & (data.evaluation_region == region) & (data.method == method)]
            agg = sub.groupby("n_train")["R2"].agg(["mean", "std"]).reindex(budgets)
            ax.errorbar(budgets, agg["mean"], yerr=agg["std"], color=COLORS[method], marker=MARKERS[method], lw=1.6, ms=4, capsize=2, label=method)
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(x) for x in budgets])
        ax.set_title(region.capitalize())
        ax.set_xlabel("Label budget $N$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Temperature $R^2$")
    axes[0].legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "test1_r2_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "test1_r2_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 9.0), sharex=True)
    for row, test in enumerate(["Test 2", "Test 3", "Test 4"]):
        for col, region in enumerate(["interpolation", "extrapolation"]):
            ax = axes[row, col]
            for method in METHODS:
                sub = data[(data.test == test) & (data.evaluation_region == region) & (data.method == method)]
                agg = sub.groupby("n_train")["Swat_R2"].agg(["mean", "std"]).reindex(budgets)
                ax.errorbar(budgets, agg["mean"], yerr=agg["std"], color=COLORS[method], marker=MARKERS[method], lw=1.5, ms=4, capsize=2, label=method)
            ax.set_xscale("log")
            ax.set_xticks(budgets, [str(x) for x in budgets])
            ax.set_title(f"{test}: {region}")
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel("Water saturation $R^2$")
            if row == 2:
                ax.set_xlabel("Label budget $N$")
    axes[0, 0].legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "swat_r2_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "swat_r2_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_advantage_map(data: pd.DataFrame) -> None:
    budgets = [20, 50, 100, 200, 500, 1000]
    rows, labels = [], []
    for test in [f"Test {i}" for i in range(1, 5)]:
        metric = metric_for(test, "R2")
        for region in ["interpolation", "extrapolation"]:
            values = []
            for budget in budgets:
                stats = summary(data, test, region, budget, metric)
                values.append(stats.loc[["H-PINN", "HO-PINN"], "mean"].max() - stats.loc[["XGB", "SVM", "GP"], "mean"].max())
            rows.append(values)
            labels.append(f"{test} {'interp.' if region == 'interpolation' else 'extrap.'}")
    matrix = np.asarray(rows)
    limit = np.max(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(budgets)), budgets)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Label budget $N$")
    ax.set_title("Best PINN minus best classical $R^2$")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.45 * limit else "black")
    fig.colorbar(image, ax=ax, shrink=0.88, label="$\\Delta R^2$")
    fig.tight_layout()
    fig.savefig(FIGURES / "pinn_advantage_map.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "pinn_advantage_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def copy_qualitative_figures() -> None:
    source = PAPER / "figures" / "final_model_comparison"
    selections = {
        "qualitative_test1.png": source / "test1" / "test1_n20_s101_profiles.png",
        "qualitative_test2.png": source / "test2" / "test2_n50_s101_t0.3.png",
        "qualitative_test3.png": source / "test3" / "test3_n50_s101_t0.05.png",
        "qualitative_test4.png": source / "test4" / "test4_n20_s101_t0.2_sample007.png",
    }
    for name, path in selections.items():
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copyfile(path, FIGURES / name)


def write_audit(data: pd.DataFrame) -> None:
    data.sort_values(["test", "evaluation_region", "n_train", "seed", "method"]).to_csv(
        OUT / "final_results_used.csv", index=False
    )
    lines = ["test,region,N,best_pinn,best_pinn_R2,best_classical,best_classical_R2,delta_R2"]
    for test in [f"Test {i}" for i in range(1, 5)]:
        metric = metric_for(test, "R2")
        for region in ["interpolation", "extrapolation"]:
            for budget in [20, 50, 100, 200, 500, 1000]:
                stats = summary(data, test, region, budget, metric)
                pinn = stats.loc[["H-PINN", "HO-PINN"], "mean"].idxmax()
                classical = stats.loc[["XGB", "SVM", "GP"], "mean"].idxmax()
                p = stats.loc[pinn, "mean"]
                c = stats.loc[classical, "mean"]
                lines.append(f"{test},{region},{budget},{pinn},{p:.8f},{classical},{c:.8f},{p-c:.8f}")
    (OUT / "best_method_comparison.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_results()
    write_primary_table(data)
    write_flow_tables(data)
    write_runtime_table(data)
    write_secondary_metric_table(data, "Pressure")
    write_secondary_metric_table(data, "Swat")
    write_physical_diagnostics_table(data)
    write_test1_secondary_table(data)
    plot_scaling(data)
    plot_advantage_map(data)
    copy_qualitative_figures()
    write_audit(data)
    print(f"Built article assets from {len(data)} final result rows in {ARTICLE}")


if __name__ == "__main__":
    main()
