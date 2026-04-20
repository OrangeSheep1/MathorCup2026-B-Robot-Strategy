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
    policy_table = policy_table[
        (policy_table["recovery_lock"] == 0)
        & (policy_table["counter_ready"] == 0)
        & (policy_table["health_opp"] == 2)
    ].copy()
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
    figure.suptitle("Q3 最优策略分布矩阵（对手 High）", fontproperties=zh_font, fontsize=TITLE_SIZE - 1, y=0.975)
    figure.supxlabel("分差", fontproperties=zh_font, fontsize=LABEL_SIZE, y=0.040)
    figure.supylabel("宏观动作类", fontproperties=zh_font, fontsize=LABEL_SIZE, x=0.035)

    output = Path(output_path)
    figure.subplots_adjust(left=0.14, right=0.875, bottom=0.08, top=0.925, wspace=0.06, hspace=0.14)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _compress_action_segments(trajectory_table: pd.DataFrame) -> pd.DataFrame:
    """将连续相同动作压缩为单个区段，便于轨迹图展示。"""

    if trajectory_table.empty:
        return pd.DataFrame()

    segments: list[dict[str, object]] = []
    for method, subset in trajectory_table.sort_values("time_step").groupby("method"):
        current_segment: dict[str, object] | None = None
        for row in subset.to_dict("records"):
            time_step = int(row["time_step"])
            action_id = str(row["action_id"])
            macro_group = str(row["macro_group"])
            if (
                current_segment is not None
                and action_id == current_segment["action_id"]
                and macro_group == current_segment["macro_group"]
                and time_step == int(current_segment["end_step"]) + 1
            ):
                current_segment["end_step"] = time_step
                current_segment["length"] = int(current_segment["length"]) + 1
                continue

            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {
                "method": method,
                "action_id": action_id,
                "macro_group": macro_group,
                "start_step": time_step,
                "end_step": time_step,
                "length": 1,
            }
        if current_segment is not None:
            segments.append(current_segment)

    segment_table = pd.DataFrame(segments)
    if segment_table.empty:
        return segment_table
    segment_table["label"] = segment_table.apply(
        lambda row: f"{row['action_id']}×{int(row['length'])}" if int(row["length"]) > 1 else str(row["action_id"]),
        axis=1,
    )
    segment_table["x_center"] = (segment_table["start_step"] + segment_table["end_step"]) / 2.0
    segment_table["width"] = segment_table["end_step"] - segment_table["start_step"] + 0.90
    return segment_table


def plot_value_surface(policy_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制值函数二维填色等高图。"""

    zh_font, _ = configure_fonts()
    subset = policy_table[
        (policy_table["health_my"] == 2)
        & (policy_table["health_opp"] == 2)
        & (policy_table["recovery_lock"] == 0)
        & (policy_table["counter_ready"] == 0)
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
    figure, ax = plt.subplots(figsize=(7.0, 4.2))
    methods = ["greedy", "static", "rule", "mdp"]
    bar_width = 0.18
    x_centers = np.arange(len(SCENARIO_ORDER), dtype=float)
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(methods))
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    upper_bound = 1.0

    for offset, method in zip(offsets, methods, strict=True):
        subset = metrics_table[metrics_table["method"] == method].set_index("scenario").reindex(SCENARIO_ORDER)
        heights = subset["win_rate"].to_numpy(dtype=float)
        ci_low = subset["ci_low"].to_numpy(dtype=float)
        ci_high = subset["ci_high"].to_numpy(dtype=float)
        upper_bound = max(upper_bound, float(np.nanmax(ci_high)))
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
            label_x = bar.get_x() + bar.get_width() / 2.0
            if height >= 0.88:
                label_y = max(height - 0.05, 0.03)
                label_color = "white"
                va = "top"
            else:
                label_y = upper + 0.012
                label_color = "#111827"
                va = "bottom"
            ax.text(
                label_x,
                label_y,
                f"{height:.2f}",
                ha="center",
                va=va,
                fontsize=TEXT_SIZE,
                color=label_color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.10}
                if label_color != "white"
                else None,
                zorder=5,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(SCENARIO_ORDER, fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax.set_ylabel("胜率", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax.set_ylim(0.0, min(1.18, upper_bound + 0.10))
    ax.set_title("Q3 四方法胜率对比", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, pad=8)
    _set_axis_style(ax)
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        ncol=4,
        prop=zh_font,
    )

    output = Path(output_path)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.82)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_trajectory_comparison(trajectory_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制样例对局轨迹对比图。"""

    zh_font, _ = configure_fonts()
    methods = ["greedy", "mdp"]
    if not trajectory_table.empty and trajectory_table["scenario"].nunique() > 1:
        return plot_trajectory_scenarios(trajectory_table, output_path)

    figure = plt.figure(figsize=(7.1, 4.9))
    grid = figure.add_gridspec(2, 1, height_ratios=[2.15, 0.62], hspace=0.20)
    ax_top = figure.add_subplot(grid[0, 0])
    ax_bottom = figure.add_subplot(grid[1, 0])
    if trajectory_table.empty:
        output = Path(output_path)
        figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output

    scenario_name = str(trajectory_table["scenario"].iloc[0])
    sample_seed = int(trajectory_table["match_seed"].iloc[0])
    segment_table = _compress_action_segments(trajectory_table)

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
            linestyle="--" if method == "greedy" else "-",
            label=METHOD_LABELS[method],
        )
        ax_top.plot(
            subset["time_step"],
            subset["score_diff_after"],
            marker="s" if method == "greedy" else "o",
            markersize=4.2,
            linestyle="none",
            color=METHOD_COLORS[method],
            markerfacecolor="white" if method == "greedy" else METHOD_COLORS[method],
            markeredgewidth=1.0,
        )

    ax_top.axhline(0.0, color="#A0AEC0", linewidth=0.9, linestyle="--")
    ax_top.set_xlim(0.35, max(float(trajectory_table["time_step"].max()), 1.0) + 0.35)
    ax_top.set_ylabel("分差", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax_top.set_title(
        f"Q3 轨迹对比：MDP vs 贪心（{scenario_name}，种子 {sample_seed}）",
        fontproperties=zh_font,
        fontsize=TITLE_SIZE,
        pad=6,
    )
    _set_axis_style(ax_top)
    ax_top.legend(frameon=False, fontsize=TEXT_SIZE + 1, loc="upper left", prop=zh_font)

    y_positions = {"mdp": 1.0, "greedy": 0.0}
    for method in methods:
        subset = segment_table[segment_table["method"] == method].copy()
        if subset.empty:
            continue
        y_level = y_positions[method]
        for _, row in subset.iterrows():
            rect = Rectangle(
                (float(row["x_center"]) - float(row["width"]) / 2.0, y_level - 0.15),
                float(row["width"]),
                0.30,
                facecolor=MACRO_COLORS.get(str(row["macro_group"]), "#CBD5E0"),
                edgecolor="#1A202C",
                linewidth=0.45,
            )
            ax_bottom.add_patch(rect)
            ax_bottom.text(
                float(row["x_center"]),
                y_level,
                str(row["label"]),
                ha="center",
                va="center",
                fontsize=SMALL_TEXT_SIZE + 1,
                color="white" if row["macro_group"] in {"激进攻击", "反击防守", "平衡恢复"} else "#111827",
                weight="bold",
                clip_on=False,
            )

    ax_bottom.set_xlim(0.35, max(float(trajectory_table["time_step"].max()), 1.0) + 0.35)
    ax_bottom.set_ylim(-0.42, 1.42)
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_yticklabels(["贪心", "MDP"], fontproperties=zh_font, fontsize=LABEL_SIZE)
    _set_axis_style(ax_bottom)
    figure.supxlabel("时间步", fontproperties=zh_font, fontsize=LABEL_SIZE + 1, y=0.070)

    legend_handles = [
        Line2D([0], [0], color=color, lw=6, label=label)
        for label, color in MACRO_COLORS.items()
    ]
    figure.legend(
        legend_handles,
        [item.get_label() for item in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        frameon=False,
        ncol=5,
        prop=zh_font,
    )

    output = Path(output_path)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.91, hspace=0.08)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_trajectory_scenarios(trajectory_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制三类场景代表轨迹，避免单一样本误导。"""

    zh_font, _ = configure_fonts()
    methods = ["greedy", "mdp"]
    scenarios = [item for item in SCENARIO_ORDER if item in set(trajectory_table["scenario"])]
    figure = plt.figure(figsize=(9.6, 8.6))
    grid = figure.add_gridspec(
        len(scenarios),
        2,
        width_ratios=[1.22, 1.05],
        left=0.075,
        right=0.975,
        bottom=0.17,
        top=0.92,
        hspace=0.46,
        wspace=0.20,
    )

    for row_index, scenario in enumerate(scenarios):
        local = trajectory_table[trajectory_table["scenario"] == scenario].copy()
        ax_score = figure.add_subplot(grid[row_index, 0])
        ax_action = figure.add_subplot(grid[row_index, 1])

        for method in methods:
            subset = local[local["method"] == method].copy()
            if subset.empty:
                continue
            ax_score.step(
                subset["time_step"],
                subset["score_diff_after"],
                where="post",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                linestyle="--" if method == "greedy" else "-",
                label=METHOD_LABELS[method],
            )
            ax_score.plot(
                subset["time_step"],
                subset["score_diff_after"],
                marker="s" if method == "greedy" else "o",
                markersize=3.3,
                linestyle="none",
                color=METHOD_COLORS[method],
                markerfacecolor="white" if method == "greedy" else METHOD_COLORS[method],
                markeredgewidth=0.8,
            )

        ax_score.axhline(0.0, color="#A0AEC0", linewidth=0.85, linestyle="--")
        ax_score.set_title(f"{scenario} 分差轨迹", fontproperties=zh_font, fontsize=LABEL_SIZE + 1, pad=4)
        ax_score.set_xlim(0.35, max(float(local["time_step"].max()), 1.0) + 0.35)
        ax_score.set_ylabel("分差", fontproperties=zh_font, fontsize=LABEL_SIZE)
        if row_index == len(scenarios) - 1:
            ax_score.set_xlabel("时间步", fontproperties=zh_font, fontsize=LABEL_SIZE)
        else:
            ax_score.tick_params(axis="x", labelbottom=False)
        _set_axis_style(ax_score)
        if row_index == 0:
            ax_score.legend(frameon=False, prop=zh_font, loc="upper left")

        segment_table = _compress_action_segments(local)
        y_positions = {"mdp": 1.0, "greedy": 0.0}
        for method in methods:
            subset = segment_table[segment_table["method"] == method].copy()
            for _, seg in subset.iterrows():
                rect = Rectangle(
                    (float(seg["x_center"]) - float(seg["width"]) / 2.0, y_positions[method] - 0.16),
                    float(seg["width"]),
                    0.32,
                    facecolor=MACRO_COLORS.get(str(seg["macro_group"]), "#CBD5E0"),
                    edgecolor="#1A202C",
                    linewidth=0.42,
                )
                ax_action.add_patch(rect)
                if float(seg["width"]) >= 1.5:
                    ax_action.text(
                        float(seg["x_center"]),
                        y_positions[method],
                        str(seg["label"]),
                        ha="center",
                        va="center",
                        fontsize=SMALL_TEXT_SIZE,
                        color="white" if seg["macro_group"] in {"激进攻击", "反击防守", "平衡恢复"} else "#111827",
                        weight="bold",
                        clip_on=True,
                    )
        ax_action.set_xlim(0.35, max(float(local["time_step"].max()), 1.0) + 0.35)
        ax_action.set_ylim(-0.45, 1.45)
        ax_action.set_yticks([0.0, 1.0])
        ax_action.set_yticklabels(["贪心", "MDP"], fontproperties=zh_font, fontsize=LABEL_SIZE)
        ax_action.set_title(f"{scenario} 动作片段", fontproperties=zh_font, fontsize=LABEL_SIZE + 1, pad=4)
        if row_index == len(scenarios) - 1:
            ax_action.set_xlabel("时间步", fontproperties=zh_font, fontsize=LABEL_SIZE)
        else:
            ax_action.tick_params(axis="x", labelbottom=False)
        _set_axis_style(ax_action)

    legend_handles = [Line2D([0], [0], color=color, lw=6, label=label) for label, color in MACRO_COLORS.items()]
    figure.legend(
        legend_handles,
        [item.get_label() for item in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        ncol=5,
        prop=zh_font,
    )
    figure.suptitle("Q3 三场景代表轨迹：MDP vs 贪心", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, y=0.985)
    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_scenario_strategy_main(
    policy_table: pd.DataFrame,
    process_metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制 Q3 主图：场景动作占比 + 关键过程指标。"""

    zh_font, _ = configure_fonts()
    base = policy_table[(policy_table["recovery_lock"] == 0) & (policy_table["counter_ready"] == 0)].copy()
    scenario_masks = {
        "领先局": (base["score_diff"] >= 2) & (base["time_step"] >= 16),
        "平局局": (base["score_diff"] == 0) & (base["time_step"].between(10, 14)),
        "落后局": (base["score_diff"] <= -2) & (base["time_step"] >= 16),
    }

    share_rows: list[dict[str, object]] = []
    for scenario, mask in scenario_masks.items():
        subset = base[mask].copy()
        counts = subset["mdp_macro_group"].value_counts(normalize=True)
        for macro in MACRO_ORDER:
            share_rows.append({"scenario": scenario, "macro_group": macro, "share": float(counts.get(macro, 0.0))})
    share_table = pd.DataFrame(share_rows)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax_left, ax_right = axes

    bottoms = np.zeros(len(SCENARIO_ORDER), dtype=float)
    x_values = np.arange(len(SCENARIO_ORDER), dtype=float)
    for macro in MACRO_ORDER:
        heights = (
            share_table[share_table["macro_group"] == macro]
            .set_index("scenario")
            .reindex(SCENARIO_ORDER)["share"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        ax_left.bar(
            x_values,
            heights,
            bottom=bottoms,
            color=MACRO_COLORS[macro],
            edgecolor="#1A202C",
            linewidth=0.45,
            label=macro,
        )
        for x, bottom, height in zip(x_values, bottoms, heights, strict=True):
            if height >= 0.08:
                ax_left.text(
                    x,
                    bottom + height / 2.0,
                    f"{height:.0%}",
                    ha="center",
                    va="center",
                    color="white" if macro in {"激进攻击", "反击防守", "平衡恢复"} else "#111827",
                    fontsize=SMALL_TEXT_SIZE,
                    weight="bold",
                )
        bottoms += heights
    ax_left.set_xticks(x_values)
    ax_left.set_xticklabels(SCENARIO_ORDER, fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("MDP 动作占比", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_title("三类场景最优策略结构", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=6)
    _set_axis_style(ax_left)

    mdp_metrics = process_metrics[process_metrics["method"] == "mdp"].set_index("scenario").reindex(SCENARIO_ORDER)
    greedy_metrics = process_metrics[process_metrics["method"] == "greedy"].set_index("scenario").reindex(SCENARIO_ORDER)
    metrics = [
        ("defense_use_rate", "防守使用率"),
        ("late_lead_defense_rate", "领先后期防守"),
        ("late_trail_aggressive_rate", "落后后期激攻"),
    ]
    bar_width = 0.12
    metric_x = np.arange(len(metrics), dtype=float)
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        offset = (scenario_index - 1) * bar_width
        mdp_values = [float(mdp_metrics.loc[scenario, col]) if not pd.isna(mdp_metrics.loc[scenario, col]) else 0.0 for col, _ in metrics]
        ax_right.bar(
            metric_x + offset,
            mdp_values,
            width=bar_width,
            color=["#2F855A", "#2B6CB0", "#C53030"][scenario_index],
            edgecolor="#1A202C",
            linewidth=0.45,
            label=scenario,
        )
    ax_right.plot(
        metric_x,
        [
            float(greedy_metrics[col].mean(skipna=True))
            for col, _ in metrics
        ],
        color="#111827",
        marker="o",
        linewidth=1.8,
        label="贪心均值",
    )
    ax_right.set_xticks(metric_x)
    ax_right.set_xticklabels([label for _, label in metrics], fontproperties=zh_font, fontsize=SMALL_TEXT_SIZE + 1)
    ax_right.set_ylim(0.0, 1.05)
    ax_right.set_ylabel("比例", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_right.set_title("关键过程指标（MDP）", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=6)
    _set_axis_style(ax_right)

    handles, labels = ax_left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.34, 0.005), frameon=False, ncol=5, prop=zh_font)
    ax_right.legend(frameon=False, prop=zh_font, fontsize=SMALL_TEXT_SIZE, loc="upper right")
    output = Path(output_path)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.88, wspace=0.24)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_process_metrics(process_metrics: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制过程指标对比图，作为胜率图的补充。"""

    zh_font, _ = configure_fonts()
    metrics = [
        ("defense_use_rate", "防守使用率"),
        ("late_lead_defense_rate", "领先后期防守"),
        ("late_trail_aggressive_rate", "落后后期激攻"),
        ("attack_defense_switch_rate", "攻防切换率"),
    ]
    methods = ["greedy", "static", "rule", "mdp"]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharey=False)
    axes_flat = axes.ravel()

    for ax, (column, title) in zip(axes_flat, metrics, strict=True):
        x_centers = np.arange(len(SCENARIO_ORDER), dtype=float)
        bar_width = 0.18
        offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(methods))
        for offset, method in zip(offsets, methods, strict=True):
            subset = process_metrics[process_metrics["method"] == method].set_index("scenario").reindex(SCENARIO_ORDER)
            values = subset[column].astype(float).fillna(0.0).to_numpy()
            ax.bar(
                x_centers + offset,
                values,
                width=bar_width,
                color=METHOD_COLORS[method],
                edgecolor="#1A202C",
                linewidth=0.35,
                label=METHOD_LABELS[method],
            )
        ax.set_title(title, fontproperties=zh_font, fontsize=LABEL_SIZE + 1, pad=5)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(SCENARIO_ORDER, fontproperties=zh_font, fontsize=SMALL_TEXT_SIZE + 1)
        ax.set_ylim(0.0, 1.05)
        _set_axis_style(ax)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), frameon=False, ncol=4, prop=zh_font)
    figure.suptitle("Q3 过程指标对比：MDP 优势不只体现在胜率", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, y=1.035)
    output = Path(output_path)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.88, hspace=0.32, wspace=0.18)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_composite_score(composite_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制 Q3 综合评价图，作为胜率图之外的主结论支撑。"""

    zh_font, _ = configure_fonts()
    methods = ["greedy", "static", "rule", "mdp"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax_left, ax_right = axes
    x_centers = np.arange(len(SCENARIO_ORDER), dtype=float)
    bar_width = 0.18
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(methods))

    for offset, method in zip(offsets, methods, strict=True):
        subset = composite_table[composite_table["method"] == method].set_index("scenario").reindex(SCENARIO_ORDER)
        value_column = "composite_index" if "composite_index" in subset.columns else "composite_score"
        values = subset[value_column].astype(float).fillna(0.0).to_numpy()
        bars = ax_left.bar(
            x_centers + offset,
            values,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="#1A202C",
            linewidth=0.4,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            ax_left.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=SMALL_TEXT_SIZE,
                color="#111827",
                zorder=4,
            )

    ax_left.set_xticks(x_centers)
    ax_left.set_xticklabels(SCENARIO_ORDER, fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_ylim(0.0, 1.08)
    ax_left.set_ylabel("综合表现指数", fontproperties=zh_font, fontsize=LABEL_SIZE + 1)
    ax_left.set_title("三类场景综合表现", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=6)
    _set_axis_style(ax_left)

    mdp = composite_table[composite_table["method"] == "mdp"].set_index("scenario").reindex(SCENARIO_ORDER)
    components = [
        ("win_rate", "胜率", 0.45),
        ("score_diff_norm", "分差", 0.20),
        ("lead_defense_component", "领先稳态", 0.20),
        ("trail_attack_component", "落后追分", 0.10),
        ("switch_component", "切换", 0.05),
    ]
    bottom = np.zeros(len(SCENARIO_ORDER), dtype=float)
    component_colors = ["#2B6CB0", "#2F855A", "#805AD5", "#C53030", "#DD6B20"]
    for (column, label, weight), color in zip(components, component_colors, strict=True):
        values = mdp[column].astype(float).fillna(0.0).to_numpy() * weight
        ax_right.bar(
            x_centers,
            values,
            bottom=bottom,
            width=0.48,
            color=color,
            edgecolor="#1A202C",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
        bottom += values

    ax_right.set_xticks(x_centers)
    ax_right.set_xticklabels(SCENARIO_ORDER, fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_right.set_ylim(0.0, 1.08)
    ax_right.set_title("MDP 得分构成", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=6)
    _set_axis_style(ax_right)
    ax_right.legend(frameon=False, prop=zh_font, fontsize=SMALL_TEXT_SIZE, loc="upper right")

    handles, labels = ax_left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.38, 1.02), frameon=False, ncol=4, prop=zh_font)
    figure.suptitle("Q3 综合表现指数：胜率、分差与过程适配性", fontproperties=zh_font, fontsize=TITLE_SIZE + 1, y=1.08)
    output = Path(output_path)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.82, wspace=0.18)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_qgap_analysis(qgap_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制 A04 Q-gap 诊断图。"""

    zh_font, _ = configure_fonts()
    table = qgap_table.copy()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    ax_left, ax_right = axes

    x_values = np.arange(len(table), dtype=float)
    ax_left.bar(
        x_values,
        table["a04_top1_share"].astype(float),
        color="#C53030",
        edgecolor="#1A202C",
        linewidth=0.45,
    )
    ax_left.set_xticks(x_values)
    ax_left.set_xticklabels(table["scope_label"], rotation=28, ha="right", fontproperties=zh_font, fontsize=SMALL_TEXT_SIZE)
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_ylabel("A04 Top1 占比", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_title("A04 是否高频入选", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=6)
    _set_axis_style(ax_left)

    width = 0.34
    ax_right.bar(
        x_values - width / 2,
        table["mean_q_gap_when_a04_top1"].astype(float),
        width=width,
        color="#2B6CB0",
        edgecolor="#1A202C",
        linewidth=0.45,
        label="平均 Q-gap",
    )
    ax_right.bar(
        x_values + width / 2,
        table["large_gap_share"].astype(float),
        width=width,
        color="#805AD5",
        edgecolor="#1A202C",
        linewidth=0.45,
        label="大间隔占比",
    )
    ax_right.set_xticks(x_values)
    ax_right.set_xticklabels(table["scope_label"], rotation=28, ha="right", fontproperties=zh_font, fontsize=SMALL_TEXT_SIZE)
    ax_right.set_title("A04 是略优还是显著强", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=6)
    ax_right.legend(frameon=False, prop=zh_font)
    _set_axis_style(ax_right)

    output = Path(output_path)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.26, top=0.86, wspace=0.22)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
