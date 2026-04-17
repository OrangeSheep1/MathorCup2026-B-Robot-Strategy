"""Q1 图表输出模块。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


FIGURE_DPI = 520
TITLE_SIZE = 18
LABEL_SIZE = 14
TICK_SIZE = 12
TEXT_SIZE = 12

CATEGORY_COLORS = {
    "punch": "#285EAD",
    "kick": "#2F855A",
    "combo": "#B7791F",
    "special": "#C53030",
}


def _pick_font(candidates: list[str], fallback: str) -> str:
    """从本机可用字体中选择目标字体。"""

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return fallback


def configure_fonts() -> tuple[FontProperties, FontProperties]:
    """设置中文宋体与英文新罗马字体。"""

    chinese_name = _pick_font(["SimSun", "NSimSun", "Songti SC", "STSong"], "DejaVu Sans")
    english_name = _pick_font(
        ["Times New Roman", "Times New Roman PS MT", "Nimbus Roman", "DejaVu Serif"],
        "DejaVu Serif",
    )
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [english_name, "DejaVu Serif"]
    mpl.rcParams["axes.unicode_minus"] = False
    return FontProperties(family=chinese_name), FontProperties(family=english_name)


def _set_axis_style(ax: plt.Axes) -> None:
    """统一坐标轴风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(linestyle="--", alpha=0.16)
    ax.tick_params(labelsize=TICK_SIZE)


def _set_chinese_axis_labels(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    zh_font: FontProperties,
) -> None:
    """设置中文标题与坐标轴标签。"""

    ax.set_title(title, fontproperties=zh_font, fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel(xlabel, fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontproperties=zh_font, fontsize=LABEL_SIZE)


def _smooth_segment(x0: float, x1: float, y0: float, y1: float, points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """生成平滑曲线段。"""

    t = np.linspace(0.0, 1.0, points)
    x = x0 + (x1 - x0) * t
    smooth = 3 * t**2 - 2 * t**3
    y = y0 + (y1 - y0) * smooth
    return x, y


def plot_utility_bar(data: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制四种方法综合对比主图。"""

    zh_font, _ = configure_fonts()
    ordered = data.sort_values(by="utility", ascending=True).copy()
    top_actions = ordered.tail(8).copy()

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 6.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax_left, ax_right = axes

    methods = [
        ("method1_utility", "方法一", "#90CDF4"),
        ("method2_utility", "方法二", "#68D391"),
        ("method3_utility", "方法三", "#F6AD55"),
        ("utility", "方法四", "#805AD5"),
    ]
    y_positions = np.arange(len(ordered))
    offsets = [-0.24, -0.08, 0.08, 0.24]
    for offset, (column, label, color) in zip(offsets, methods, strict=True):
        ax_left.barh(
            y_positions + offset,
            ordered[column],
            height=0.14,
            color=color,
            edgecolor="#1A202C",
            linewidth=0.4,
            label=label,
        )
    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels(ordered["action_id"] + " " + ordered["action_name"])
    ax_left.set_xlim(0.0, 1.08)
    _set_axis_style(ax_left)
    _set_chinese_axis_labels(ax_left, "四种方法归一化得分对比", "归一化得分", "攻击动作", zh_font)
    ax_left.legend(prop=zh_font, frameon=False, loc="lower right")

    rank_columns = ["method1_rank", "method2_rank", "method3_rank", "rank"]
    rank_labels = ["方法一", "方法二", "方法三", "方法四"]
    x_positions = np.arange(len(rank_columns))
    for _, row in top_actions.iterrows():
        scores = [len(ordered) + 1 - int(row[column]) for column in rank_columns]
        ax_right.plot(
            x_positions,
            scores,
            color=CATEGORY_COLORS.get(row["category"], "#4A5568"),
            linewidth=2.4,
            alpha=0.92,
            marker="o",
            markersize=6,
        )
        ax_right.text(
            x_positions[-1] + 0.08,
            scores[-1],
            row["action_id"],
            fontsize=TEXT_SIZE - 1,
            va="center",
        )
    ax_right.set_xticks(x_positions)
    ax_right.set_xticklabels(rank_labels, fontproperties=zh_font)
    ax_right.set_yticks(np.arange(1, len(ordered) + 1))
    ax_right.set_yticklabels([str(value) for value in range(len(ordered), 0, -1)])
    _set_axis_style(ax_right)
    _set_chinese_axis_labels(ax_right, "Top 8 动作跨方法排名轨迹", "评价方法", "名次优势", zh_font)
    figure.tight_layout()

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_impact_balance_scatter(data: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制攻击效果与平衡代价决策散点图。"""

    zh_font, _ = configure_fonts()
    figure, ax = plt.subplots(figsize=(10.6, 7.2))
    safe_margin = float(data["stable_margin"].iloc[0])

    ax.axhspan(0.0, safe_margin, color="#E6FFFA", alpha=0.65)
    ax.axhspan(safe_margin, float(data["balance_cost"].max()) * 1.15, color="#FFF5F5", alpha=0.55)
    ax.axhline(safe_margin, color="#C53030", linestyle="--", linewidth=1.6)
    ax.text(
        float(data["impact_score"].min()),
        safe_margin + 0.004,
        f"稳定裕度阈值 = {safe_margin:.3f} m",
        fontproperties=zh_font,
        fontsize=TEXT_SIZE,
        color="#9B2C2C",
    )

    pareto = data.sort_values(by="impact_score")
    scatter = ax.scatter(
        data["impact_score"],
        data["balance_cost"],
        s=320 * data["score_prob"] + 40,
        c=data["utility"],
        cmap="viridis",
        edgecolors="#1A202C",
        linewidths=0.8,
        alpha=0.92,
        zorder=3,
    )
    ax.plot(pareto["impact_score"], pareto["balance_cost"], linestyle=":", color="#4A5568", linewidth=1.1, alpha=0.65)
    for _, row in data.iterrows():
        ax.annotate(
            row["action_id"],
            (row["impact_score"], row["balance_cost"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=TEXT_SIZE - 2,
            weight="bold",
        )

    _set_axis_style(ax)
    _set_chinese_axis_labels(ax, "攻击效果与平衡代价权衡图", "冲击力矩 / N·m", "质心偏移 / m", zh_font)
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("归一化效用", fontproperties=zh_font, fontsize=LABEL_SIZE)
    figure.tight_layout()

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_method_comparison(data: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制四种方法排名流图。"""

    zh_font, _ = configure_fonts()
    methods = [
        ("method1_rank", "方法一"),
        ("method2_rank", "方法二"),
        ("method3_rank", "方法三"),
        ("rank", "方法四"),
    ]
    x_positions = np.arange(len(methods))
    color_map = plt.get_cmap("viridis")
    normalized = colors.Normalize(vmin=float(data["utility"].min()), vmax=float(data["utility"].max()))

    figure, ax = plt.subplots(figsize=(11.4, 7.2))
    n_actions = len(data)
    y_maps: dict[str, dict[str, float]] = {}
    for rank_column, _ in methods:
        ordered = data.sort_values(by=[rank_column, "action_id"], ascending=[True, True]).reset_index(drop=True)
        y_maps[rank_column] = {
            row["action_id"]: float(n_actions - idx)
            for idx, (_, row) in enumerate(ordered.iterrows())
        }

    for _, row in data.iterrows():
        xs = []
        ys = []
        for method_idx in range(len(methods) - 1):
            rank_column_left = methods[method_idx][0]
            rank_column_right = methods[method_idx + 1][0]
            x_segment, y_segment = _smooth_segment(
                x_positions[method_idx],
                x_positions[method_idx + 1],
                y_maps[rank_column_left][row["action_id"]],
                y_maps[rank_column_right][row["action_id"]],
            )
            if method_idx > 0:
                x_segment = x_segment[1:]
                y_segment = y_segment[1:]
            xs.extend(x_segment.tolist())
            ys.extend(y_segment.tolist())

        line_color = color_map(normalized(float(row["utility"])))
        linewidth = 1.4 + 2.8 * float(row["utility"])
        alpha = 0.25 + 0.65 * float(row["utility"])
        ax.plot(xs, ys, color=line_color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")

    for method_idx, (rank_column, label) in enumerate(methods):
        ax.scatter(
            np.full(n_actions, x_positions[method_idx]),
            [y_maps[rank_column][action_id] for action_id in data["action_id"]],
            s=22,
            color="#1A202C",
            zorder=4,
        )
        ax.text(
            x_positions[method_idx],
            n_actions + 1.0,
            label,
            ha="center",
            fontproperties=zh_font,
            fontsize=LABEL_SIZE,
            weight="bold",
        )

    top5 = data.nlargest(5, "utility")
    for _, row in top5.iterrows():
        ax.text(
            x_positions[-1] + 0.08,
            y_maps["rank"][row["action_id"]],
            f"{row['action_id']} {row['action_name']}",
            va="center",
            fontsize=TEXT_SIZE - 1,
            color="#1A202C",
        )

    ax.set_xlim(-0.2, x_positions[-1] + 1.1)
    ax.set_ylim(0.5, n_actions + 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("四种方法排名流图", fontproperties=zh_font, fontsize=TITLE_SIZE, pad=14)
    figure.tight_layout()

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_penalty_curve(data: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制非线性倒地惩罚曲线。"""

    zh_font, _ = configure_fonts()
    figure, ax = plt.subplots(figsize=(10.0, 6.8))
    stable_margin = float(data["stable_margin"].iloc[0])
    x_values = np.linspace(0.0, float(data["balance_cost"].max()) * 1.08, 600)
    lambda_value = 0.30
    k_value = 3.50
    y_values = lambda_value * np.exp(k_value * x_values / stable_margin) - lambda_value

    ax.semilogy(x_values, y_values, color="#6B46C1", linewidth=2.3)
    ax.axvline(stable_margin, color="#C53030", linestyle="--", linewidth=1.6)
    ax.text(
        stable_margin + 0.002,
        y_values.max() / 60.0,
        "稳定裕度阈值",
        fontproperties=zh_font,
        fontsize=TEXT_SIZE,
        color="#9B2C2C",
    )

    category_colors = [CATEGORY_COLORS.get(category, "#4A5568") for category in data["category"]]
    ax.scatter(
        data["balance_cost"],
        data["fall_penalty_raw"],
        color=category_colors,
        s=95,
        edgecolors="#1A202C",
        linewidths=0.8,
        zorder=3,
    )
    for _, row in data.iterrows():
        ax.annotate(
            row["action_id"],
            (row["balance_cost"], row["fall_penalty_raw"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=TEXT_SIZE - 2,
        )

    _set_axis_style(ax)
    _set_chinese_axis_labels(ax, "非线性倒地惩罚函数与动作落点", "质心偏移 / m", "原始惩罚项（对数尺度）", zh_font)
    figure.tight_layout()

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_decision_atlas(data: pd.DataFrame, output_path: str | Path) -> Path:
    """绘制方法四决策图谱。"""

    zh_font, _ = configure_fonts()
    figure, ax = plt.subplots(figsize=(10.6, 7.2))

    u0_max = float(data["u0"].max())
    penalty_raw_max = float(data["fall_penalty_raw"].max())
    ratio_grid = np.linspace(0.0, max(2.0, float(data["stability_ratio"].max()) * 1.08), 240)
    u0_grid = np.linspace(float(data["u0"].min()) - 0.05, u0_max + 0.05, 220)
    x_mesh, y_mesh = np.meshgrid(ratio_grid, u0_grid)
    raw_penalty = 0.30 * np.exp(3.50 * x_mesh) - 0.30
    normalized_penalty = raw_penalty / penalty_raw_max * u0_max
    score_surface = np.maximum(0.0, y_mesh - normalized_penalty)

    contour = ax.contourf(x_mesh, y_mesh, score_surface, levels=18, cmap="YlGnBu")
    contour_lines = ax.contour(x_mesh, y_mesh, score_surface, levels=8, colors="white", linewidths=0.8, alpha=0.65)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
    ax.axvline(1.0, color="#C53030", linestyle="--", linewidth=1.6)
    ax.text(1.02, float(data["u0"].min()) + 0.02, "稳定裕度临界线", fontproperties=zh_font, fontsize=TEXT_SIZE, color="#9B2C2C")

    ax.scatter(
        data["stability_ratio"],
        data["u0"],
        s=220,
        c=data["utility"],
        cmap="viridis",
        edgecolors="#1A202C",
        linewidths=0.9,
        zorder=4,
    )
    for _, row in data.iterrows():
        ax.annotate(
            row["action_id"],
            (row["stability_ratio"], row["u0"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=TEXT_SIZE - 2,
            weight="bold",
        )

    _set_axis_style(ax)
    _set_chinese_axis_labels(ax, "方法四决策图谱", "稳定性比值 ΔCoM / θstable", "基础效用 U0", zh_font)
    colorbar = figure.colorbar(contour, ax=ax, pad=0.02)
    colorbar.set_label("最终得分等高面", fontproperties=zh_font, fontsize=LABEL_SIZE)
    figure.tight_layout()

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_sensitivity_heatmap(
    sensitivity: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """绘制灵敏度分析热力图。"""

    zh_font, _ = configure_fonts()
    overlap = sensitivity["top3_overlap"]
    spinning = sensitivity["spinning_utility"]

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 5.8))
    images = []
    matrices = [overlap, spinning]
    titles = ["Top 3 保持率", "回旋踢效用"]

    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        image = ax.imshow(matrix.values, cmap="YlGnBu", aspect="auto")
        images.append(image)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, fontsize=TICK_SIZE)
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=TICK_SIZE)
        ax.set_title(title, fontproperties=zh_font, fontsize=LABEL_SIZE, pad=10)
        ax.set_xlabel("k", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
        ax.set_ylabel("λ", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix.iloc[row_idx, col_idx]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=TEXT_SIZE - 2,
                    color="#1A202C",
                )

    colorbar_left = figure.colorbar(images[0], ax=axes[0], pad=0.02)
    colorbar_left.set_label("比例", fontproperties=zh_font, fontsize=LABEL_SIZE - 2)
    colorbar_right = figure.colorbar(images[1], ax=axes[1], pad=0.02)
    colorbar_right.set_label("效用", fontproperties=zh_font, fontsize=LABEL_SIZE - 2)
    figure.suptitle("惩罚参数灵敏度分析", fontproperties=zh_font, fontsize=TITLE_SIZE, y=0.98)
    figure.tight_layout(rect=[0, 0, 1, 0.95])

    output = Path(output_path)
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
