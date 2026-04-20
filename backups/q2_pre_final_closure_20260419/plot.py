"""Q2 图表输出模块。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import griddata

from src.q1.plot import FIGURE_DPI, configure_fonts


TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 10
TEXT_SIZE = 10

DEFENSE_COLORS = {
    "block": "#285EAD",
    "evade": "#2F855A",
    "posture": "#B7791F",
    "balance": "#805AD5",
    "ground": "#C53030",
    "combo": "#0F766E",
}

METHOD_LABELS = {
    "method1": "方法一",
    "method2": "方法二",
    "method3": "方法三",
    "method4": "方法四",
}


def _set_axis_style(ax: plt.Axes, show_grid: bool = True) -> None:
    """统一坐标轴风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    if show_grid:
        ax.grid(linestyle="--", alpha=0.14)
    ax.tick_params(labelsize=TICK_SIZE)


def _smooth_curve(x0: float, x1: float, y0: float, y1: float, points: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """生成平滑连线。"""

    t = np.linspace(0.0, 1.0, points)
    x = x0 + (x1 - x0) * t
    s = 3.0 * t**2 - 2.0 * t**3
    y = y0 + (y1 - y0) * s
    return x, y


def _build_method_top_table(evaluated_pairs: pd.DataFrame) -> pd.DataFrame:
    """抽取四种方法的 Top1 防守。"""

    method_specs = [
        ("method1", "method1_score"),
        ("method2", "method2_score"),
        ("method3", "method3_score"),
        ("method4", "proposed_score"),
    ]
    records: list[dict[str, object]] = []
    for action_id, group in evaluated_pairs.groupby("action_id", sort=False):
        record = {"action_id": action_id, "action_name": group.iloc[0]["action_name"]}
        for method_key, score_column in method_specs:
            top_row = group.sort_values(score_column, ascending=False).iloc[0]
            record[f"{method_key}_defense_id"] = top_row["defense_id"]
            record[f"{method_key}_score"] = float(top_row[score_column])
            record[f"{method_key}_category"] = top_row["defense_category"]
        records.append(record)
    return pd.DataFrame(records)


def plot_hierarchical_utility_matrix(
    evaluated_pairs: pd.DataFrame,
    counter_chain: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制简洁版聚类效用矩阵。"""

    zh_font, _ = configure_fonts()
    pivot = evaluated_pairs.pivot(index="action_id", columns="defense_id", values="proposed_score").fillna(0.0)
    row_linkage = linkage(pivot.values, method="ward", optimal_ordering=True)
    col_linkage = linkage(pivot.values.T, method="ward", optimal_ordering=True)
    row_order = dendrogram(row_linkage, no_plot=True)["leaves"]
    col_order = dendrogram(col_linkage, no_plot=True)["leaves"]
    ordered = pivot.iloc[row_order, col_order]

    row_attack_ids = ordered.index.tolist()
    col_defense_ids = ordered.columns.tolist()
    top1 = counter_chain[counter_chain["priority_rank"] == 1]

    figure, ax = plt.subplots(figsize=(7.4, 5.8))
    image = ax.imshow(ordered.values, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(col_defense_ids)))
    ax.set_xticklabels(col_defense_ids, rotation=55, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(row_attack_ids)))
    ax.set_yticklabels(row_attack_ids, fontsize=TICK_SIZE)
    ax.set_xlabel("防守动作", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("攻击动作", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 主模型攻防效用矩阵", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)

    for row_index, attack_id in enumerate(row_attack_ids):
        matched = top1[top1["action_id"] == attack_id]
        if matched.empty:
            continue
        defense_id = str(matched.iloc[0]["defense_id"])
        if defense_id not in col_defense_ids:
            continue
        col_index = col_defense_ids.index(defense_id)
        ax.scatter(col_index, row_index, marker="*", s=80, color="white", edgecolors="#111827", linewidths=0.5)

    colorbar = figure.colorbar(image, ax=ax, fraction=0.036, pad=0.02)
    colorbar.set_label("主模型得分", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_SIZE)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_defense_surface(evaluated_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制二维等高响应面图。"""

    zh_font, _ = configure_fonts()
    x = evaluated_pairs["tau_norm"].to_numpy(dtype=float)
    y = evaluated_pairs["p_fall"].to_numpy(dtype=float)
    z = evaluated_pairs["proposed_score"].to_numpy(dtype=float)

    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, 120),
        np.linspace(0.0, min(1.0, max(0.12, float(y.max()) + 0.05)), 120),
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
    fill_z = griddata((x, y), z, (grid_x, grid_y), method="nearest")
    grid_z = np.where(np.isnan(grid_z), fill_z, grid_z)

    top1 = evaluated_pairs[evaluated_pairs["rank"] == 1].copy()

    figure, ax = plt.subplots(figsize=(7.0, 5.4))
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=14, cmap="Spectral_r")
    ax.contour(grid_x, grid_y, grid_z, levels=8, colors="white", linewidths=0.45, alpha=0.55)
    ax.scatter(x, y, s=10, color="#334155", alpha=0.18)

    marker_map = {
        "block": "o",
        "evade": "^",
        "posture": "s",
        "balance": "D",
        "ground": "P",
        "combo": "X",
    }
    for defense_category, marker in marker_map.items():
        subset = top1[top1["defense_category"] == defense_category]
        if subset.empty:
            continue
        ax.scatter(
            subset["tau_norm"],
            subset["p_fall"],
            s=48,
            marker=marker,
            color=DEFENSE_COLORS.get(defense_category, "#4A5568"),
            edgecolors="#111827",
            linewidths=0.45,
            alpha=0.95,
        )

    ax.set_xlabel("攻击强度归一值", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("防守倒地风险", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 主模型防守效用响应面", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax)

    colorbar = figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("综合防守得分", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_SIZE)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[key],
            color="w",
            label=key,
            markerfacecolor=DEFENSE_COLORS[key],
            markeredgecolor="#111827",
            markeredgewidth=0.45,
            markersize=6,
        )
        for key in marker_map
        if not top1[top1["defense_category"] == key].empty
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=TEXT_SIZE - 1, ncol=2)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_decision_waterfall(
    matchup_table: pd.DataFrame,
    evaluated_pairs: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """绘制简洁版攻击到 Top1 防守映射图。"""

    zh_font, _ = configure_fonts()
    rows = []
    for _, row in matchup_table.iterrows():
        defense_id = row["defense_id_r1"]
        matched = evaluated_pairs[
            (evaluated_pairs["action_id"] == row["action_id"]) & (evaluated_pairs["defense_id"] == defense_id)
        ]
        if matched.empty:
            continue
        rows.append(matched.iloc[0])
    top1 = pd.DataFrame(rows).sort_values(by=["attack_utility", "proposed_score"], ascending=[False, False]).reset_index(drop=True)

    attack_nodes = top1["action_id"].tolist()
    defense_nodes = (
        top1.groupby("defense_id")["proposed_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    def _positions(labels: list[str], top: float = 0.92, bottom: float = 0.08) -> dict[str, float]:
        if len(labels) == 1:
            return {labels[0]: (top + bottom) / 2.0}
        values = np.linspace(top, bottom, len(labels))
        return {label: float(value) for label, value in zip(labels, values, strict=True)}

    attack_pos = _positions(attack_nodes)
    defense_pos = _positions(defense_nodes)

    figure, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    x_left = 0.18
    x_right = 0.82
    node_width = 0.12
    node_height = 0.034

    for _, row in top1.iterrows():
        x_curve, y_curve = _smooth_curve(
            x_left + node_width / 2.0,
            x_right - node_width / 2.0,
            attack_pos[str(row["action_id"])],
            defense_pos[str(row["defense_id"])],
        )
        ax.plot(
            x_curve,
            y_curve,
            color=DEFENSE_COLORS.get(str(row["defense_category"]), "#4A5568"),
            linewidth=1.0 + 3.0 * float(row["proposed_utility"]),
            alpha=0.42,
            solid_capstyle="round",
        )

    for attack_id in attack_nodes:
        rect = plt.Rectangle(
            (x_left - node_width / 2.0, attack_pos[attack_id] - node_height / 2.0),
            node_width,
            node_height,
            facecolor="#F8FAFC",
            edgecolor="#1F2937",
            linewidth=0.55,
        )
        ax.add_patch(rect)
        ax.text(x_left, attack_pos[attack_id], attack_id, ha="center", va="center", fontsize=TEXT_SIZE)

    for defense_id in defense_nodes:
        defense_category = str(top1.loc[top1["defense_id"] == defense_id, "defense_category"].iloc[0])
        rect = plt.Rectangle(
            (x_right - node_width / 2.0, defense_pos[defense_id] - node_height / 2.0),
            node_width,
            node_height,
            facecolor=DEFENSE_COLORS.get(defense_category, "#4A5568"),
            edgecolor="#1F2937",
            linewidth=0.55,
        )
        ax.add_patch(rect)
        ax.text(x_right, defense_pos[defense_id], defense_id, ha="center", va="center", fontsize=TEXT_SIZE, color="white")

    ax.text(x_left, 0.975, "攻击", ha="center", va="top", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.text(x_right, 0.975, "Top1 防守", ha="center", va="top", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 攻击到 Top1 防守映射图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_parallel_bands(evaluated_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制简洁版平行坐标图。"""

    zh_font, _ = configure_fonts()
    metric_names = ["拦截概率", "防守安全性", "反击窗口", "稳定性"]
    metric_frame = pd.DataFrame(
        {
            "拦截概率": evaluated_pairs["p_block"],
            "防守安全性": 1.0 - evaluated_pairs["defense_damage"],
            "反击窗口": evaluated_pairs["counter_window_norm"],
            "稳定性": 1.0 - evaluated_pairs["p_fall"],
        }
    )
    metric_frame["score_bin"] = pd.qcut(
        evaluated_pairs["proposed_score"].rank(method="first"),
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    x_positions = np.arange(len(metric_names))
    palette = cm.get_cmap("Spectral_r", 5)

    figure, ax = plt.subplots(figsize=(7.2, 5.3))
    for x_pos in x_positions:
        ax.axvline(x=x_pos, color="#CBD5E0", linewidth=0.8, alpha=0.85)

    for index, score_bin in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"]):
        subset = metric_frame[metric_frame["score_bin"] == score_bin]
        means = subset[metric_names].mean().to_numpy(dtype=float)
        stds = subset[metric_names].std(ddof=0).fillna(0.0).to_numpy(dtype=float)
        color = palette(index)
        ax.plot(x_positions, means, color=color, linewidth=2.4, alpha=0.96, label=score_bin)
        ax.fill_between(
            x_positions,
            np.clip(means - stds, 0.0, 1.0),
            np.clip(means + stds, 0.0, 1.0),
            color=color,
            alpha=0.12,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("归一化维度值", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 防守效用平行坐标图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax)

    legend = ax.legend(title="得分分位", frameon=False, fontsize=TEXT_SIZE - 1, loc="upper left", ncol=3)
    if legend is not None:
        legend.get_title().set_fontproperties(zh_font)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_method_comparison(evaluated_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制四方法 Top1 防守对比图。"""

    zh_font, _ = configure_fonts()
    top_table = _build_method_top_table(evaluated_pairs)
    attacks = top_table.sort_values(by="method4_score", ascending=False)["action_id"].tolist()
    methods = ["method1", "method2", "method3", "method4"]
    x_positions = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    marker_size = 240

    figure, ax = plt.subplots(figsize=(7.6, 5.2))

    for x_pos in x_positions:
        ax.axvline(x=x_pos, color="#E2E8F0", linewidth=0.9, zorder=0)

    for y_index, attack_id in enumerate(attacks):
        row = top_table.loc[top_table["action_id"] == attack_id].iloc[0]
        for x_index, method_key in enumerate(methods):
            defense_id = str(row[f"{method_key}_defense_id"])
            defense_category = str(row[f"{method_key}_category"])
            ax.scatter(
                x_positions[x_index],
                y_index,
                s=marker_size,
                color=DEFENSE_COLORS.get(defense_category, "#4A5568"),
                edgecolors="#111827",
                linewidths=0.45,
                alpha=0.96,
                zorder=3,
            )
            ax.text(
                x_positions[x_index],
                y_index,
                defense_id,
                ha="center",
                va="center",
                fontsize=TEXT_SIZE - 1,
                color="white",
                weight="bold",
                zorder=4,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[item] for item in methods], fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=LABEL_SIZE - 1)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 3.55)
    ax.set_xlabel("评价方法", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("攻击动作", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 四方法 Top1 防守对比图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=8)
    _set_axis_style(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=key,
            markerfacecolor=value,
            markeredgecolor="#111827",
            markeredgewidth=0.45,
            markersize=6,
        )
        for key, value in DEFENSE_COLORS.items()
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
        fontsize=TEXT_SIZE - 1,
    )

    output = Path(output_path)
    figure.subplots_adjust(top=0.88, bottom=0.18, left=0.14, right=0.98)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
