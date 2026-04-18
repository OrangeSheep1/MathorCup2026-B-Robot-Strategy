"""Q3 图表输出模块。"""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.q1.plot import FIGURE_DPI, configure_fonts
from src.q3.model_v1 import HEALTH_LABELS, SCORE_DIFF_VALUES, TIME_PHASE_LABELS


TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 10
TEXT_SIZE = 10
SMALL_TEXT_SIZE = 9
HEALTH_SHORT_LABELS = {
    "High": "高",
    "Mid": "中",
    "Low": "低",
}
MACRO_SHORT_LABELS = {
    "激进攻击": "激攻",
    "试探攻击": "试攻",
    "反击防守": "反守",
    "保守防守": "保守",
    "平衡恢复": "恢复",
}

MACRO_COLORS = {
    "激进攻击": "#C53030",
    "试探攻击": "#DD6B20",
    "反击防守": "#2B6CB0",
    "保守防守": "#2F855A",
    "平衡恢复": "#805AD5",
}

METHOD_COLORS = {
    "greedy": "#DD6B20",
    "static": "#718096",
    "rule": "#2F855A",
    "mdp": "#2B6CB0",
}

METHOD_LABELS = {
    "greedy": "贪心",
    "static": "静态博弈",
    "rule": "规则树",
    "mdp": "MDP",
}

SCENARIO_ORDER = ["领先局", "平局局", "落后局"]
PHASE_ORDER = ["early", "mid", "late", "endgame"]
MACRO_ORDER = ["激进攻击", "试探攻击", "反击防守", "保守防守", "平衡恢复"]


def _set_axis_style(ax: plt.Axes, show_grid: bool = True) -> None:
    """统一坐标轴风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    if show_grid:
        ax.grid(linestyle="--", alpha=0.14)
    ax.tick_params(labelsize=TICK_SIZE)


def plot_policy_heatmap(policy_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制最优策略分布矩阵。"""

    zh_font, _ = configure_fonts()
    policy_table = policy_table[policy_table["recovery_lock"] == 0].copy()
    figure, axes = plt.subplots(4, 3, figsize=(10.4, 8.0), sharex=True, sharey=True)
    cmap = LinearSegmentedColormap.from_list(
        "q3_policy",
        ["#F8FAFC", "#C6F6D5", "#63B3ED", "#F6AD55", "#E53E3E"],
    )
    last_image = None

    for row_index, phase in enumerate(PHASE_ORDER):
        for col_index, health_my in enumerate([2, 1, 0]):
            ax = axes[row_index, col_index]
            subset = policy_table[
                (policy_table["time_phase"] == phase)
                & (policy_table["health_my"] == health_my)
            ].copy()
            matrix = np.zeros((len(MACRO_ORDER), len(SCORE_DIFF_VALUES)), dtype=float)
            for x_index, score_diff in enumerate(SCORE_DIFF_VALUES):
                local = subset[subset["score_diff"] == score_diff]
                if local.empty:
                    continue
                counts = local["mdp_macro_group"].value_counts(normalize=True)
                for y_index, macro_group in enumerate(MACRO_ORDER):
                    matrix[y_index, x_index] = float(counts.get(macro_group, 0.0))

            last_image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(
                f"{TIME_PHASE_LABELS[phase]}/{HEALTH_SHORT_LABELS[HEALTH_LABELS[health_my]]}",
                fontproperties=zh_font,
                fontsize=LABEL_SIZE,
                pad=4,
            )
            if row_index == len(PHASE_ORDER) - 1:
                ax.set_xticks(np.arange(len(SCORE_DIFF_VALUES)))
                ax.set_xticklabels([str(value) for value in SCORE_DIFF_VALUES], fontsize=TICK_SIZE)
            else:
                ax.set_xticks(np.arange(len(SCORE_DIFF_VALUES)))
                ax.tick_params(axis="x", labelbottom=False)
            if col_index == 0:
                ax.set_yticks(np.arange(len(MACRO_ORDER)))
                ax.set_yticklabels(
                    [MACRO_SHORT_LABELS[item] for item in MACRO_ORDER],
                    fontproperties=zh_font,
                    fontsize=SMALL_TEXT_SIZE + 1,
                )
            else:
                ax.set_yticks(np.arange(len(MACRO_ORDER)))
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="y", pad=3)
            _set_axis_style(ax, show_grid=False)

    cax = figure.add_axes([0.888, 0.20, 0.010, 0.58])
    colorbar = figure.colorbar(last_image, cax=cax)
    colorbar.set_label("占比", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_SIZE - 1)
    figure.suptitle("Q3 最优策略分布矩阵", fontproperties=zh_font, fontsize=TITLE_SIZE - 1, y=0.980)
    figure.supxlabel("分差", fontproperties=zh_font, fontsize=LABEL_SIZE, y=0.040)
    figure.supylabel("宏观动作类", fontproperties=zh_font, fontsize=LABEL_SIZE, x=0.035)

    output = Path(output_path)
    figure.subplots_adjust(left=0.14, right=0.875, bottom=0.08, top=0.925, wspace=0.06, hspace=0.14)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_value_surface(policy_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制值函数二维填色等高图。"""

    zh_font, _ = configure_fonts()
    subset = policy_table[
        (policy_table["health_my"] == 2)
        & (policy_table["health_opp"] == 2)
        & (policy_table["recovery_lock"] == 0)
    ].copy()
    pivot = subset.pivot(index="score_diff", columns="time_step", values="mdp_value").sort_index()
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_mesh = pivot.to_numpy(dtype=float)

    figure, ax = plt.subplots(figsize=(7.4, 5.3))
    filled = ax.contourf(
        x_mesh,
        y_mesh,
        z_mesh,
        levels=12,
        cmap="Spectral_r",
    )
    contours = ax.contour(
        x_mesh,
        y_mesh,
        z_mesh,
        levels=8,
        colors="#364152",
        linewidths=0.75,
        alpha=0.55,
    )
    ax.clabel(contours, inline=True, fmt="%.1f", fontsize=SMALL_TEXT_SIZE - 1)
    gradient = np.gradient(z_mesh)
    importance = np.abs(gradient[0]) + np.abs(gradient[1])
    important_indices = np.dstack(np.unravel_index(np.argsort(importance.ravel())[-4:], importance.shape))[0]
    for row_index, col_index in important_indices:
        ax.scatter(
            x_mesh[row_index, col_index],
            y_mesh[row_index, col_index],
            color="#0F172A",
            edgecolors="white",
            linewidths=0.7,
            s=36,
            zorder=4,
        )

    ax.set_xlabel("时间步", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax.set_ylabel("分差", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([-5, -3, -1, 1, 3, 5])
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("Q3 值函数等高图（High / High）", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, pad=8)
    _set_axis_style(ax, show_grid=False)

    cax = figure.add_axes([0.88, 0.18, 0.018, 0.58])
    colorbar = figure.colorbar(filled, cax=cax)
    colorbar.set_label("值函数", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_SIZE - 1)

    output = Path(output_path)
    figure.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.90)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_method_comparison(metrics_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制四方法胜率对比图。"""

    zh_font, _ = configure_fonts()
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    methods = ["greedy", "static", "rule", "mdp"]
    bar_width = 0.18
    x_centers = np.arange(len(SCENARIO_ORDER), dtype=float)
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(methods))
    legend_handles: list[object] = []
    legend_labels: list[str] = []

    for offset, method in zip(offsets, methods, strict=True):
        subset = metrics_table[metrics_table["method"] == method].set_index("scenario").reindex(SCENARIO_ORDER)
        heights = subset["win_rate"].to_numpy(dtype=float)
        ci_low = subset["ci_low"].to_numpy(dtype=float)
        ci_high = subset["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([heights - ci_low, ci_high - heights])
        bars = ax.bar(
            x_centers + offset,
            heights,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="#1A202C",
            linewidth=0.45,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        legend_handles.append(bars[0])
        legend_labels.append(METHOD_LABELS[method])
        ax.errorbar(
            x_centers + offset,
            heights,
            yerr=yerr,
            fmt="none",
            ecolor="#1A202C",
            elinewidth=0.8,
            capsize=3,
            zorder=4,
        )
        for bar, height, upper in zip(bars, heights, ci_high, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                upper + 0.008,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=TEXT_SIZE,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 0.15,
                },
                zorder=5,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(SCENARIO_ORDER, fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax.set_ylabel("胜率", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax.set_ylim(0.0, 1.08)
    ax.set_title("Q3 四方法胜率对比", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, pad=8)
    _set_axis_style(ax)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        prop=zh_font,
    )

    output = Path(output_path)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.90)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_trajectory_comparison(trajectory_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制样例对局轨迹对比图。"""

    zh_font, _ = configure_fonts()
    methods = ["greedy", "mdp"]
    figure = plt.figure(figsize=(7.4, 5.5))
    grid = figure.add_gridspec(2, 1, height_ratios=[2.0, 0.84], hspace=0.14)
    ax_top = figure.add_subplot(grid[0, 0])
    ax_bottom = figure.add_subplot(grid[1, 0])

    for method in methods:
        subset = trajectory_table[trajectory_table["method"] == method].copy()
        if subset.empty:
            continue
        ax_top.step(
            subset["time_step"],
            subset["score_diff_after"],
            where="post",
            linewidth=2.3,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        ax_top.plot(
            subset["time_step"],
            subset["score_diff_after"],
            marker="o",
            markersize=4.2,
            linestyle="none",
            color=METHOD_COLORS[method],
        )

    ax_top.axhline(0.0, color="#A0AEC0", linewidth=0.9, linestyle="--")
    ax_top.set_xlim(0.35, max(float(trajectory_table["time_step"].max()), 1.0) + 0.35)
    ax_top.set_ylabel("分差", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax_top.set_title("Q3 轨迹对比：MDP vs 贪心", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, pad=6)
    _set_axis_style(ax_top)
    ax_top.legend(frameon=False, fontsize=TEXT_SIZE + 1, loc="upper left", prop=zh_font)

    y_positions = {"mdp": 1.0, "greedy": 0.0}
    for method in methods:
        subset = trajectory_table[trajectory_table["method"] == method].copy()
        if subset.empty:
            continue
        y_level = y_positions[method]
        for _, row in subset.iterrows():
            rect = Rectangle(
                (float(row["time_step"]) - 0.40, y_level - 0.14),
                0.80,
                0.28,
                facecolor=MACRO_COLORS.get(str(row["macro_group"]), "#CBD5E0"),
                edgecolor="#1A202C",
                linewidth=0.45,
            )
            ax_bottom.add_patch(rect)
            ax_bottom.text(
                float(row["time_step"]),
                y_level,
                str(row["action_id"]),
                ha="center",
                va="center",
                fontsize=SMALL_TEXT_SIZE + 0.5,
                color="white" if row["macro_group"] in {"激进攻击", "反击防守", "平衡恢复"} else "#111827",
                weight="bold",
                clip_on=False,
            )

    ax_bottom.set_xlim(0.35, max(float(trajectory_table["time_step"].max()), 1.0) + 0.35)
    ax_bottom.set_ylim(-0.42, 1.42)
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_yticklabels(["贪心", "MDP"], fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_bottom.set_xlabel("时间步", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    _set_axis_style(ax_bottom)

    legend_handles = [
        Line2D([0], [0], color=color, lw=6, label=label)
        for label, color in MACRO_COLORS.items()
    ]
    figure.legend(
        legend_handles,
        [item.get_label() for item in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.002),
        frameon=False,
        ncol=5,
        prop=zh_font,
    )

    output = Path(output_path)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.92, hspace=0.14)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
