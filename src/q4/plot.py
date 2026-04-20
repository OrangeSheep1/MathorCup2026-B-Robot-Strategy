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
SCENARIO_COLORS = {
    "leading": "#2B6CB0",
    "tied": "#2F855A",
    "trailing": "#C53030",
}
ACTION_COLORS = {
    "试探攻击": "#63B3ED",
    "激进攻击": "#C53030",
    "反击防守": "#805AD5",
    "保守防守": "#2B6CB0",
    "平衡恢复": "#2F855A",
    "人工复位": "#D69E2E",
    "战术暂停": "#38A169",
    "紧急维修": "#DD6B20",
    "倒地等待": "#718096",
    "故障等待": "#4A5568",
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
        expected_usage = f"用R{float(row['expected_used_reset']):.1f}/P{float(row['expected_used_pause']):.1f}/M{float(row['expected_used_repair']):.1f}"
        ax.text(
            x0,
            y0 - 0.02,
            str(row["allocation_label"]),
            ha="center",
            va="center",
            fontsize=TEXT_SIZE - 1,
            color="#1A202C",
            zorder=4,
        )
        ax.text(
            x0,
            y0 - 0.22,
            expected_usage,
            ha="center",
            va="center",
            fontsize=TEXT_SIZE - 2,
            fontproperties=zh_font,
            color="#2D3748",
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
                f"{label}\nP={float(row['round_pwin']):.2f}" if linestyle == "-" else label,
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

    ax.set_title("Q4 BO3 动态资源策略树", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
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


def plot_resource_gain(resource_uplift: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制三场景零资源与最优资源胜率增益主图。"""

    zh_font, _ = configure_fonts()
    order = ["leading", "tied", "trailing"]
    labels = ["领先局", "平局局", "落后局"]
    best_rows = (
        resource_uplift.sort_values(["scenario_key", "p_win"], ascending=[True, False])
        .groupby("scenario_key", as_index=False)
        .head(1)
        .set_index("scenario_key")
    )
    zero_rows = (
        resource_uplift[resource_uplift["allocation_label"] == "R0-P0-M0"]
        .set_index("scenario_key")
    )
    zero_values = [float(zero_rows.loc[key, "p_win"]) for key in order]
    best_values = [float(best_rows.loc[key, "p_win"]) for key in order]
    best_labels = [str(best_rows.loc[key, "allocation_label"]) for key in order]

    x = np.arange(len(order))
    width = 0.34
    figure, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(x - width / 2, zero_values, width, label="零资源", color="#A0AEC0", edgecolor="#1A202C", linewidth=0.7)
    ax.bar(x + width / 2, best_values, width, label="最优资源", color="#2B6CB0", edgecolor="#1A202C", linewidth=0.7)
    for index, (value, label) in enumerate(zip(best_values, best_labels, strict=True)):
        ax.text(index + width / 2, value + 0.015, label, ha="center", va="bottom", fontsize=TEXT_SIZE, fontproperties=zh_font)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=zh_font, fontsize=TICK_SIZE)
    ax.set_ylabel("单局获胜概率", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q4 三场景资源增益主图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    ax.set_ylim(0.0, min(1.05, max(best_values + zero_values) + 0.12))
    ax.legend(frameon=False, prop=zh_font)
    _set_axis_style(ax, grid=True)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_resource_timing(
    resource_usage: pd.DataFrame,
    first_use_distribution: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制资源使用率与首次使用中位时机。"""

    zh_font, _ = configure_fonts()
    order = ["leading", "tied", "trailing"]
    labels = ["领先局", "平局局", "落后局"]
    usage = resource_usage.set_index("scenario_key")
    x = np.arange(len(order))
    width = 0.22
    figure, ax_left = plt.subplots(figsize=(9.2, 5.3))
    ax_right = ax_left.twinx()

    bars = [
        ("reset_use_rate", "复位率", "#63B3ED", -width),
        ("pause_use_rate", "暂停率", "#2F855A", 0.0),
        ("repair_use_rate", "维修率", "#DD6B20", width),
    ]
    for column, label, color, offset in bars:
        ax_left.bar(
            x + offset,
            [float(usage.loc[key, column]) for key in order],
            width,
            label=label,
            color=color,
            edgecolor="#1A202C",
            linewidth=0.6,
            alpha=0.84,
        )

    median_rows = first_use_distribution.pivot_table(
        index="scenario_key",
        columns="resource_type",
        values="median_min",
        aggfunc="first",
    )
    for resource_type, label, color, marker in [
        ("reset", "复位中位时机", "#2B6CB0", "o"),
        ("pause", "暂停中位时机", "#276749", "s"),
        ("repair", "维修中位时机", "#C05621", "^"),
    ]:
        if resource_type not in median_rows.columns:
            continue
        values = [float(median_rows.loc[key, resource_type]) if key in median_rows.index and pd.notna(median_rows.loc[key, resource_type]) else np.nan for key in order]
        ax_right.plot(x, values, color=color, marker=marker, linewidth=1.8, label=label)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontproperties=zh_font, fontsize=TICK_SIZE)
    ax_left.set_ylabel("资源使用率", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_right.set_ylabel("首次使用中位时刻 / 分钟", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_title("Q4 三场景资源使用率与时机", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    _set_axis_style(ax_left, grid=True)
    ax_right.spines["top"].set_visible(False)
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, prop=zh_font)

    output = Path(output_path)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_resource_policy_heatmap(policy_heatmap_table: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制状态条件下的资源/战术推荐热图。"""

    zh_font, _ = configure_fonts()
    if policy_heatmap_table.empty:
        figure, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.text(0.5, 0.5, "无资源策略热图数据", ha="center", va="center", fontproperties=zh_font)
        ax.axis("off")
        output = Path(output_path)
        figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output

    score_order = ["大幅领先", "小幅领先", "平局", "小幅落后", "大幅落后"]
    time_order = ["前期", "中期", "后期"]
    status_order = ["正常", "低机能", "故障", "倒地"]
    row_labels = [f"{score}\n{phase}" for score in score_order for phase in time_order]
    row_keys = [(score, phase) for score in score_order for phase in time_order]
    action_names = sorted(policy_heatmap_table["recommended_action_name"].dropna().unique().tolist())
    action_to_code = {name: index + 1 for index, name in enumerate(action_names)}
    code_to_color = {0: "#F7FAFC"}
    for name, code in action_to_code.items():
        code_to_color[code] = ACTION_COLORS.get(name, "#A0AEC0")

    matrix = np.zeros((len(row_keys), len(status_order)), dtype=float)
    annotation = [["" for _ in status_order] for _ in row_keys]
    share_text = [["" for _ in status_order] for _ in row_keys]
    lookup = {
        (str(row["score_diff_band"]), str(row["time_bucket_band"]), str(row["status_label"])): row
        for _, row in policy_heatmap_table.iterrows()
    }
    for row_index, (score, phase) in enumerate(row_keys):
        for col_index, status in enumerate(status_order):
            row = lookup.get((score, phase, status))
            if row is None:
                continue
            action_name = str(row["recommended_action_name"])
            matrix[row_index, col_index] = action_to_code.get(action_name, 0)
            annotation[row_index][col_index] = action_name.replace("紧急", "")
            share_text[row_index][col_index] = f"{float(row['state_share']):.0%}"

    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = [code_to_color[index] for index in range(len(action_names) + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(action_names) + 1.5, 1.0), cmap.N)

    figure, ax = plt.subplots(figsize=(9.2, 8.4))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(status_order)))
    ax.set_xticklabels(status_order, fontproperties=zh_font, fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontproperties=zh_font, fontsize=TICK_SIZE - 1)
    ax.set_title("Q4 状态条件下的资源与战术推荐", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    ax.tick_params(length=0)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if annotation[row_index][col_index]:
                ax.text(
                    col_index,
                    row_index - 0.12,
                    annotation[row_index][col_index],
                    ha="center",
                    va="center",
                    fontproperties=zh_font,
                    fontsize=TEXT_SIZE - 2,
                    color="#1A202C",
                )
                ax.text(
                    col_index,
                    row_index + 0.20,
                    share_text[row_index][col_index],
                    ha="center",
                    va="center",
                    fontsize=TEXT_SIZE - 3,
                    color="#2D3748",
                )
    ax.set_xticks(np.arange(-0.5, len(status_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=ACTION_COLORS.get(name, "#A0AEC0"), markeredgecolor="#1A202C", label=name)
        for name in action_names
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=False,
        prop=zh_font,
    )
    output = Path(output_path)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
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


def plot_composite_score(composite_score: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制 Q4 综合资源调度指数。"""

    zh_font, _ = configure_fonts()
    ordered = composite_score.sort_values("composite_score", ascending=True).reset_index(drop=True)
    figure, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = [METHOD_COLORS.get(method, "#4A5568") for method in ordered["method"]]
    y_positions = np.arange(len(ordered))
    bars = ax.barh(
        y_positions,
        ordered["composite_score"],
        color=colors,
        edgecolor="#1A202C",
        linewidth=0.7,
        alpha=0.88,
    )
    for bar, value in zip(bars, ordered["composite_score"], strict=True):
        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=TEXT_SIZE,
        )
    ax.set_xlim(0.0, min(1.05, float(ordered["composite_score"].max()) + 0.10))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["method_label"], fontproperties=zh_font, fontsize=TICK_SIZE)
    ax.set_xlabel("综合资源调度指数", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q4 综合表现指数", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=10)
    _set_axis_style(ax, grid=True)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    note = "指数综合 BO3 胜率、平局转化、落后韧性和领先保守性；不是原始胜率。"
    ax.text(
        0.0,
        -0.18,
        note,
        transform=ax.transAxes,
        fontsize=TEXT_SIZE - 1,
        fontproperties=zh_font,
        color="#4A5568",
    )

    output = Path(output_path)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_main_summary(
    resource_uplift: pd.DataFrame,
    method_summary: pd.DataFrame,
    resource_usage: pd.DataFrame,
    first_use_distribution: pd.DataFrame,
    composite_score: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制 Q4 主结论总览图。"""

    zh_font, _ = configure_fonts()
    order = ["leading", "tied", "trailing"]
    scenario_labels = ["领先局", "平局局", "落后局"]
    figure = plt.figure(figsize=(13.5, 8.6))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.04, 1.0], hspace=0.34, wspace=0.24)
    ax_gain = figure.add_subplot(grid[0, 0])
    ax_method = figure.add_subplot(grid[0, 1])
    ax_usage = figure.add_subplot(grid[1, 0])
    ax_score = figure.add_subplot(grid[1, 1])

    best_rows = (
        resource_uplift.sort_values(["scenario_key", "p_win"], ascending=[True, False])
        .groupby("scenario_key", as_index=False)
        .head(1)
        .set_index("scenario_key")
    )
    zero_rows = resource_uplift[resource_uplift["allocation_label"] == "R0-P0-M0"].set_index("scenario_key")
    y_positions = np.arange(len(order))
    for index, key in enumerate(order):
        zero_value = float(zero_rows.loc[key, "p_win"])
        best_value = float(best_rows.loc[key, "p_win"])
        color = SCENARIO_COLORS[key]
        ax_gain.plot([zero_value, best_value], [index, index], color=color, linewidth=3.0, alpha=0.78)
        ax_gain.scatter(zero_value, index, s=90, color="#E2E8F0", edgecolor="#1A202C", zorder=3)
        ax_gain.scatter(best_value, index, s=120, color=color, edgecolor="#1A202C", zorder=4)
        ax_gain.text(best_value + 0.018, index, str(best_rows.loc[key, "allocation_label"]), va="center", fontsize=TEXT_SIZE, fontproperties=zh_font)
    ax_gain.set_yticks(y_positions)
    ax_gain.set_yticklabels(scenario_labels, fontproperties=zh_font)
    ax_gain.set_xlim(0.0, 1.05)
    ax_gain.set_xlabel("单局获胜概率", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_gain.set_title("资源投入带来的场景增益", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax_gain, grid=True)
    ax_gain.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#E2E8F0", markeredgecolor="#1A202C", label="零资源"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2B6CB0", markeredgecolor="#1A202C", label="最优资源"),
        ],
        frameon=False,
        prop=zh_font,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        borderaxespad=0.0,
    )

    methods = method_summary.sort_values("series_win_rate", ascending=True)
    y = np.arange(len(methods))
    ax_method.barh(
        y,
        methods["series_win_rate"],
        xerr=[
            methods["series_win_rate"] - methods["ci_low"],
            methods["ci_high"] - methods["series_win_rate"],
        ],
        color=[METHOD_COLORS.get(method, "#4A5568") for method in methods["method"]],
        edgecolor="#1A202C",
        linewidth=0.6,
        alpha=0.88,
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#1A202C"},
    )
    ax_method.set_yticks(y)
    ax_method.set_yticklabels(methods["method_label"], fontproperties=zh_font)
    ax_method.set_xlim(0.35, max(0.72, float(methods["ci_high"].max()) + 0.03))
    ax_method.set_xlabel("BO3 胜率及 95% CI", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_method.set_title("动态 DP 相对基线的优势", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax_method, grid=True)

    usage = resource_usage.set_index("scenario_key")
    resource_cols = [
        ("reset_use_rate", "复位", "#63B3ED"),
        ("pause_use_rate", "暂停", "#2F855A"),
        ("repair_use_rate", "维修", "#DD6B20"),
    ]
    x = np.arange(len(order))
    width = 0.22
    for offset, (column, label, color) in zip([-width, 0.0, width], resource_cols, strict=True):
        ax_usage.bar(
            x + offset,
            [float(usage.loc[key, column]) for key in order],
            width,
            label=label,
            color=color,
            edgecolor="#1A202C",
            linewidth=0.55,
            alpha=0.86,
        )
    median_pause = first_use_distribution[
        (first_use_distribution["resource_type"] == "pause")
    ].set_index("scenario_key")
    pause_medians = [
        float(median_pause.loc[key, "median_min"])
        if key in median_pause.index and pd.notna(median_pause.loc[key, "median_min"])
        else np.nan
        for key in order
    ]
    for index, median_value in enumerate(pause_medians):
        if pd.isna(median_value):
            label = "暂停: 未用"
        else:
            label = f"暂停中位 {median_value:.1f} 分"
        ax_usage.text(
            index,
            -0.12,
            label,
            ha="center",
            va="top",
            fontsize=TEXT_SIZE - 1,
            fontproperties=zh_font,
            color="#1A202C",
            bbox=dict(facecolor="white", edgecolor="#CBD5E0", boxstyle="round,pad=0.25", alpha=0.96),
            clip_on=False,
        )
    ax_usage.set_xticks(x)
    ax_usage.set_xticklabels(scenario_labels, fontproperties=zh_font)
    ax_usage.set_ylim(0.0, 1.05)
    ax_usage.set_ylabel("实际使用率", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_usage.set_title("资源使用强度与时机", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax_usage, grid=True)
    ax_usage.legend(loc="upper left", frameon=False, prop=zh_font, ncol=3)

    score_ordered = composite_score.sort_values("composite_score", ascending=True)
    score_y = np.arange(len(score_ordered))
    ax_score.barh(
        score_y,
        score_ordered["composite_score"],
        color=[METHOD_COLORS.get(method, "#4A5568") for method in score_ordered["method"]],
        edgecolor="#1A202C",
        linewidth=0.6,
        alpha=0.88,
    )
    for pos, value in enumerate(score_ordered["composite_score"]):
        ax_score.text(value + 0.012, pos, f"{value:.3f}", va="center", fontsize=TEXT_SIZE)
    ax_score.set_xlim(0.0, min(1.05, float(score_ordered["composite_score"].max()) + 0.10))
    ax_score.set_yticks(score_y)
    ax_score.set_yticklabels(score_ordered["method_label"], fontproperties=zh_font, fontsize=TICK_SIZE)
    ax_score.set_xlabel("综合资源调度指数", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax_score.set_title("综合表现不是单一胜率", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax_score, grid=True)

    figure.suptitle("Q4 BO3 资源调度主结论", fontproperties=zh_font, fontsize=18, y=0.985)
    output = Path(output_path)
    figure.subplots_adjust(bottom=0.15)
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
