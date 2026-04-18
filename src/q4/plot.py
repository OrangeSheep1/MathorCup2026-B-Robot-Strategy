"""Q4 图表输出模块。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.q1.plot import FIGURE_DPI, configure_fonts


TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 10
TEXT_SIZE = 10

METHOD_COLORS = {
    "optimal_dp": "#2B6CB0",
    "fixed_rule": "#DD6B20",
    "exhaustive_static": "#4A5568",
    "all_in_first": "#2F855A",
}


def _set_axis_style(ax: plt.Axes, grid: bool = True) -> None:
    """统一坐标轴风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    if grid:
        ax.grid(linestyle="--", alpha=0.15)
    ax.tick_params(labelsize=TICK_SIZE)


def plot_policy_tree(macro_policy_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制 BO3 资源分配策略树。"""

    zh_font, _ = configure_fonts()
    representative = macro_policy_table[
        (macro_policy_table["reset_left"] == 2)
        & (macro_policy_table["pause_left"] == 2)
        & (macro_policy_table["repair_left"] == 1)
    ].copy()
    positions = {
        (0, 0): (0.0, 0.0),
        (1, 0): (1.2, 1.2),
        (0, 1): (1.2, -1.2),
        (1, 1): (2.4, 0.0),
        (2, 0): (3.6, 1.2),
        (0, 2): (3.6, -1.2),
        (2, 1): (3.6, 0.4),
        (1, 2): (3.6, -0.4),
    }

    figure, ax = plt.subplots(figsize=(9.2, 5.6))
    cmap = plt.cm.RdYlGn
    value_min = representative["round_pwin"].min() if not representative.empty else 0.0
    value_max = representative["round_pwin"].max() if not representative.empty else 1.0

    for _, row in representative.iterrows():
        state = (int(row["wins_my"]), int(row["wins_opp"]))
        x0, y0 = positions[state]
        color = cmap((float(row["round_pwin"]) - value_min) / max(value_max - value_min, 1e-6))
        ax.scatter(x0, y0, s=2000, color=color, edgecolor="#1A202C", linewidth=1.0, zorder=3)
        ax.text(
            x0,
            y0 + 0.12,
            f"({state[0]},{state[1]})",
            ha="center",
            va="center",
            fontproperties=zh_font,
            fontsize=TEXT_SIZE,
            color="#1A202C",
            zorder=4,
        )
        ax.text(
            x0,
            y0 - 0.10,
            str(row["allocation_label"]),
            ha="center",
            va="center",
            fontsize=TEXT_SIZE - 1,
            color="#1A202C",
            zorder=4,
        )
        win_state = tuple(int(value) for value in row["next_win_state"].strip("()").split(","))
        lose_state = tuple(int(value) for value in row["next_lose_state"].strip("()").split(","))
        for next_state, label, linestyle in [
            (win_state, f"胜 {float(row['round_pwin']):.2f}", "-"),
            (lose_state, f"负 {1.0 - float(row['round_pwin']):.2f}", "--"),
        ]:
            if next_state not in positions:
                continue
            x1, y1 = positions[next_state]
            ax.plot([x0, x1], [y0, y1], color="#4A5568", linewidth=1.2, linestyle=linestyle, zorder=2)
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 + (0.12 if linestyle == "-" else -0.12),
                label,
                fontsize=TEXT_SIZE - 1,
                fontproperties=zh_font,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
            )

    for terminal_state in [(2, 0), (2, 1), (0, 2), (1, 2)]:
        x, y = positions[terminal_state]
        win_state = terminal_state[0] == 2
        color = "#68D391" if win_state else "#FC8181"
        ax.scatter(x, y, s=1200, color=color, edgecolor="#1A202C", linewidth=1.0, zorder=3)
        ax.text(
            x,
            y,
            f"{terminal_state}",
            ha="center",
            va="center",
            fontsize=TEXT_SIZE,
            color="#1A202C",
        )

    ax.set_title("Q4 BO3 资源分配策略树", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.4, 4.1)
    ax.set_ylim(-1.8, 1.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_pwin_heatmaps(pwin_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制三种情景下的单局胜率查表热力图。"""

    zh_font, _ = configure_fonts()
    order = ["leading", "tied", "trailing"]
    labels = [f"R{r}-P{p}-M{m}" for r in range(3) for p in range(3) for m in range(2)]
    figure, axes = plt.subplots(3, 1, figsize=(10.6, 5.9), sharex=True)
    last_image = None

    for ax, scenario_key in zip(axes, order, strict=True):
        subset = pwin_table[pwin_table["scenario_key"] == scenario_key].copy()
        subset["allocation_order"] = subset["allocation_label"].map({label: index for index, label in enumerate(labels)})
        subset = subset.sort_values("allocation_order")
        matrix = subset["p_win"].to_numpy(dtype=float).reshape(1, -1)
        last_image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=pwin_table["p_win"].min(), vmax=pwin_table["p_win"].max())
        ax.set_yticks([0])
        ax.set_yticklabels([{"leading": "领先", "tied": "平局", "trailing": "落后"}[scenario_key]], fontproperties=zh_font)
        _set_axis_style(ax, grid=False)
        ax.set_title(f"{SCENARIO_NAME_MAP[scenario_key]} 情景", fontproperties=zh_font, fontsize=LABEL_SIZE, pad=6)

    axes[-1].set_xticks(np.arange(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE - 1)
    cax = figure.add_axes([0.91, 0.18, 0.012, 0.62])
    colorbar = figure.colorbar(last_image, cax=cax)
    colorbar.set_label("单局获胜概率", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_SIZE - 1)
    figure.suptitle("Q4 单局 P_win 查找表热力图", fontproperties=zh_font, fontsize=TITLE_SIZE, y=0.97)
    figure.subplots_adjust(left=0.12, right=0.89, bottom=0.20, top=0.90, hspace=0.22)

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


SCENARIO_NAME_MAP = {
    "leading": "领先局",
    "tied": "平局局",
    "trailing": "落后局",
}


def plot_fault_curve(fault_profile: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制平均机能与故障率动态曲线。"""

    zh_font, _ = configure_fonts()
    figure, ax_left = plt.subplots(figsize=(8.1, 4.8))
    ax_right = ax_left.twinx()
    ax_left.plot(fault_profile["time_s"], fault_profile["mean_health_my"], color="#2B6CB0", linewidth=2.0, label="平均机能")
    ax_right.plot(fault_profile["time_s"], fault_profile["fault_rate"], color="#C53030", linewidth=2.0, label="故障率")
    ax_left.set_title("Q4 故障率动态曲线", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    ax_left.set_xlabel("比赛时间 / 秒", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_ylabel("平均机能档", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_right.set_ylabel("故障状态占比", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_ylim(0.0, 2.05)
    ax_right.set_ylim(0.0, max(0.1, float(fault_profile["fault_rate"].max()) * 1.2))
    _set_axis_style(ax_left, grid=True)
    ax_right.spines["top"].set_visible(False)
    handles = [
        Line2D([0], [0], color="#2B6CB0", linewidth=2.0, label="平均机能"),
        Line2D([0], [0], color="#C53030", linewidth=2.0, label="故障率"),
    ]
    ax_left.legend(handles=handles, loc="upper right", ncol=2, frameon=False, prop=zh_font)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_scenario_radar(resource_usage: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制三种情景资源使用雷达图。"""

    zh_font, _ = configure_fonts()
    metrics = [
        ("reset_use_rate", "复位率"),
        ("pause_use_rate", "暂停率"),
        ("repair_use_rate", "维修率"),
        ("avg_first_use_min", "平均时机"),
        ("single_round_win_prob", "单局胜率"),
        ("bo3_contribution", "BO3价值"),
    ]
    figure = plt.figure(figsize=(7.0, 6.2))
    ax = figure.add_subplot(111, polar=True)
    angles = np.linspace(0.0, 2.0 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    colors = {"leading": "#2B6CB0", "tied": "#DD6B20", "trailing": "#C53030"}

    for _, row in resource_usage.iterrows():
        values = [float(row[column]) for column, _ in metrics]
        values = np.array(values, dtype=float)
        scales = np.array([1.0, 1.0, 1.0, 7.0, 1.0, 1.0], dtype=float)
        values = values / scales
        values = np.concatenate([values, values[:1]])
        ax.plot(angles, values, color=colors[row["scenario_key"]], linewidth=2.0, label=row["scenario"])
        ax.fill(angles, values, color=colors[row["scenario_key"]], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in metrics], fontproperties=zh_font, fontsize=TICK_SIZE)
    ax.set_yticklabels([])
    ax.set_title("Q4 三种情景资源使用雷达图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), frameon=False, prop=zh_font)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_method_boxplot(
    batch_distribution: pd.DataFrame,
    method_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制 BO3 方法对比箱线图。"""

    zh_font, _ = configure_fonts()
    order = ["optimal_dp", "fixed_rule", "exhaustive_static", "all_in_first"]
    figure, ax = plt.subplots(figsize=(8.8, 5.2))
    data = [
        batch_distribution[batch_distribution["method"] == method]["batch_win_rate"].to_numpy(dtype=float)
        for method in order
    ]
    box = ax.boxplot(data, patch_artist=True, widths=0.58, medianprops={"color": "#1A202C", "linewidth": 1.1})
    for patch, method in zip(box["boxes"], order, strict=True):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_edgecolor("#1A202C")
        patch.set_alpha(0.78)

    label_map = method_summary.set_index("method")["method_label"].to_dict()
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([str(label_map[method]) for method in order], fontproperties=zh_font, fontsize=TICK_SIZE)
    ax.set_ylabel("批次胜率", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q4 BO3 蒙特卡洛方法对比", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    _set_axis_style(ax, grid=True)
    ax.set_ylim(0.0, 1.05)

    text_lines = []
    for method in order:
        row = method_summary[method_summary["method"] == method].iloc[0]
        text_lines.append(f"{row['method_label']}: {float(row['series_win_rate']):.3f}")
    ax.text(
        1.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=TEXT_SIZE,
        fontproperties=zh_font,
        bbox=dict(facecolor="white", edgecolor="#CBD5E0", alpha=0.95),
    )

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_tornado(sensitivity_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制灵敏度分析龙卷风图。"""

    zh_font, _ = configure_fonts()
    figure, ax = plt.subplots(figsize=(8.0, 4.8))
    ordered = sensitivity_table.sort_values(by="impact_abs", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    ax.barh(y_positions, ordered["delta_low"], color="#63B3ED", edgecolor="#1A202C", linewidth=0.5)
    ax.barh(y_positions, ordered["delta_high"], color="#FC8181", edgecolor="#1A202C", linewidth=0.5)
    ax.axvline(0.0, color="#1A202C", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["parameter_label"], fontproperties=zh_font, fontsize=TICK_SIZE)
    ax.set_xlabel("最优 BO3 胜率变化量", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q4 灵敏度分析龙卷风图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    _set_axis_style(ax, grid=True)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
