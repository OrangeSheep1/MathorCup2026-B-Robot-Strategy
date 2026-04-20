"""Q2 plotting utilities."""

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
TEXT_SIZE = 9

DEFENSE_COLORS = {
    "block": "#285EAD",
    "evade": "#2F855A",
    "posture": "#B7791F",
    "balance": "#805AD5",
    "ground": "#C53030",
    "combo": "#0F766E",
    "NA": "#94A3B8",
}

DISPLAY_EMPTY = "\u65e0"
DISPLAY_NOT_APPLICABLE = "\u2014"
NOT_APPLICABLE_COLOR = "#E2E8F0"

METHOD_LABELS = {
    "method1": "\u65b9\u6cd5\u4e00",
    "method2": "\u65b9\u6cd5\u4e8c",
    "method3": "\u65b9\u6cd5\u4e09",
    "method4": "\u65b9\u6cd5\u56db",
}

CATEGORY_LABELS = {
    "block": "\u683c\u6321",
    "evade": "\u95ea\u907f",
    "posture": "\u59ff\u6001\u7f13\u51b2",
    "balance": "\u5e73\u8861\u8c03\u6574",
    "ground": "\u5730\u9762\u9632\u62a4",
    "combo": "\u7ec4\u5408\u9632\u5b88",
    "NA": "\u65e0\u7ed3\u679c",
    "not_applicable": "\u4e0d\u9002\u7528",
}

LAYER_LABELS = {
    "active": "\u4e3b\u9632\u5b88",
    "fallback": "\u4fdd\u5e95\u54cd\u5e94",
    "ground": "\u5730\u9762\u54cd\u5e94",
    "recovery": "\u6062\u590d\u52a8\u4f5c",
}

SCATTER_LABEL_OFFSETS = {
    "A01": (0.010, -0.012),
    "A02": (-0.040, 0.010),
    "A03": (0.012, -0.014),
    "A04": (-0.045, -0.010),
    "A05": (0.012, 0.015),
    "A06": (-0.040, 0.014),
    "A07": (0.010, -0.018),
    "A08": (-0.055, -0.006),
    "A09": (0.012, 0.022),
    "A10": (0.012, -0.018),
    "A11": (-0.048, 0.012),
    "A12": (0.010, 0.010),
    "A13": (-0.048, -0.012),
}


def _set_axis_style(ax: plt.Axes, show_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_grid:
        ax.grid(linestyle="--", alpha=0.20)
    ax.tick_params(labelsize=TICK_SIZE)


def _save(figure: plt.Figure, output_path: str | Path) -> Path:
    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _set_legend_font(legend, zh_font) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontproperties(zh_font)
        text.set_fontsize(TEXT_SIZE)


def _category_legend(include_na: bool = True, include_not_applicable: bool = False) -> list[Line2D]:
    categories = ["block", "evade", "posture", "balance", "ground", "combo"]
    if include_na:
        categories.append("NA")
    if include_not_applicable:
        categories.append("not_applicable")

    handles: list[Line2D] = []
    for category in categories:
        marker_color = NOT_APPLICABLE_COLOR if category == "not_applicable" else DEFENSE_COLORS.get(category, "#94A3B8")
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=marker_color,
                markeredgecolor="#111827",
                label=CATEGORY_LABELS[category],
            )
        )
    return handles


def _legend_outside(ax: plt.Axes, figure: plt.Figure, zh_font, include_na: bool = True, include_not_applicable: bool = False) -> None:
    legend = ax.legend(
        handles=_category_legend(include_na=include_na, include_not_applicable=include_not_applicable),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=1,
    )
    _set_legend_font(legend, zh_font)
    figure.subplots_adjust(right=0.78)


def _fallen_context(row: pd.Series) -> bool:
    note = str(row.get("method4_note", "")).strip()
    return note.startswith("fallen_")


def _method_cell_state(row: pd.Series, method_key: str) -> tuple[str, str]:
    if method_key == "method4":
        defense_id = str(row.get("method4_active_top1", "")).strip()
        category = str(row.get("method4_active_category", "")).strip() or "NA"
        if defense_id == "" and _fallen_context(row):
            defense_id = str(row.get("method4_ground_top1", "")).strip()
            category = str(row.get("method4_ground_category", "")).strip() or "NA"
    else:
        defense_id = str(row.get(f"{method_key}_top1_defense", "")).strip()
        category = str(row.get(f"{method_key}_top1_category", "")).strip() or "NA"

    if defense_id == "":
        return DISPLAY_EMPTY, "NA"
    return defense_id, category if category in DEFENSE_COLORS else "NA"


def _layer_cell_state(row: pd.Series, layer_key: str, defense_column: str, category_column: str) -> tuple[str, str, str]:
    defense_id = str(row.get(defense_column, "")).strip()
    category = str(row.get(category_column, "")).strip() or "NA"
    is_fallen = _fallen_context(row)

    if layer_key == "active":
        applicable = not is_fallen
    elif layer_key == "fallback":
        applicable = not is_fallen
    elif layer_key == "ground":
        applicable = is_fallen
    elif layer_key == "recovery":
        applicable = is_fallen
    else:
        applicable = True

    if not applicable:
        return DISPLAY_NOT_APPLICABLE, "not_applicable", "not_applicable"
    if defense_id == "":
        return DISPLAY_EMPTY, "NA", "empty"
    return defense_id, category if category in DEFENSE_COLORS else "NA", "filled"


def plot_primary_utility_matrix(evaluated_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    zh_font, _ = configure_fonts()
    active_pool = evaluated_pairs[evaluated_pairs["primary_role_feasible"]].copy()
    pivot = active_pool.pivot(index="action_id", columns="defense_id", values="primary_score")
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")

    masked_values = np.ma.masked_where(pivot.isna().values, pivot.fillna(0.0).values)
    cmap = plt.cm.get_cmap("magma").copy()
    cmap.set_bad("#E2E8F0")

    figure, ax = plt.subplots(figsize=(10.5, 6.2))
    image = ax.imshow(masked_values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=60, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=TICK_SIZE)
    ax.set_xlabel("\u9632\u5b88\u52a8\u4f5c", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("\u653b\u51fb\u52a8\u4f5c", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 \u4e3b\u9632\u5b88\u5c42\u6548\u7528\u77e9\u9635", fontproperties=zh_font, fontsize=TITLE_SIZE, y=1.08)
    ax.text(
        0.0,
        1.01,
        "\u6ce8\uff1a\u4ec5\u7ad9\u7acb\u5165\u573a\u7684\u4e3b\u9632\u5b88\u5019\u9009\u8fdb\u5165\u77e9\u9635\uff1bD18/D19 \u7b49\u5730\u9762/\u6062\u590d\u52a8\u4f5c\u5c5e\u6761\u4ef6\u5c42\uff0c\u4e0d\u5728\u672c\u56fe\u8bc4\u5206\u3002",
        transform=ax.transAxes,
        fontproperties=zh_font,
        fontsize=TEXT_SIZE,
        color="#475569",
    )
    colorbar = figure.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.ax.tick_params(labelsize=TICK_SIZE)
    colorbar.set_label("\u4e3b\u9632\u5b88\u8bc4\u5206", fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    return _save(figure, output_path)


def plot_defense_surface(matchup_table: pd.DataFrame, evaluated_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    zh_font, _ = configure_fonts()
    lookup = matchup_table[["action_id", "active_top1_defense_id"]].copy()
    lookup = lookup[lookup["active_top1_defense_id"].astype(str).ne("")]
    top1 = evaluated_pairs.merge(
        lookup,
        left_on=["action_id", "defense_id"],
        right_on=["action_id", "active_top1_defense_id"],
        how="inner",
    )

    figure, ax = plt.subplots(figsize=(8.8, 5.8))
    for category, subset in top1.groupby("defense_category"):
        ax.scatter(
            subset["p_success"],
            subset["p_fall"],
            s=92,
            color=DEFENSE_COLORS.get(category, "#4A5568"),
            alpha=0.9,
            edgecolors="#111827",
            linewidths=0.5,
        )
        for _, row in subset.iterrows():
            dx, dy = SCATTER_LABEL_OFFSETS.get(str(row["action_id"]), (0.010, 0.006))
            ax.text(
                float(row["p_success"]) + dx,
                float(row["p_fall"]) + dy,
                str(row["action_id"]),
                fontsize=TEXT_SIZE,
                fontproperties=zh_font,
            )

    ax.set_xlabel("\u4e3b\u9632\u5b88\u6210\u529f\u7387", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("\u9632\u5b88\u540e\u5012\u5730\u98ce\u9669", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 \u4e3b\u9632\u5b88 Top1 \u6210\u529f\u7387\u2014\u98ce\u9669\u6563\u70b9\u56fe", fontproperties=zh_font, fontsize=TITLE_SIZE)
    _set_axis_style(ax)
    _legend_outside(ax, figure, zh_font, include_na=False)
    return _save(figure, output_path)


def plot_parallel_metrics(matchup_table: pd.DataFrame, evaluated_pairs: pd.DataFrame, output_path: str | Path) -> Path:
    zh_font, _ = configure_fonts()
    lookup = matchup_table[["action_id", "active_top1_defense_id"]].copy()
    lookup = lookup[lookup["active_top1_defense_id"].astype(str).ne("")]
    top1 = evaluated_pairs.merge(
        lookup,
        left_on=["action_id", "defense_id"],
        right_on=["action_id", "active_top1_defense_id"],
        how="inner",
    ).sort_values("action_id")

    metric_labels = ["成功率", "1-剩余伤害", "反击窗口", "1-倒地风险"]
    metric_columns = ["p_success", "defense_harm_safe", "counter_window_norm", "stability_safe"]
    x_positions = np.arange(len(metric_columns))

    figure, ax = plt.subplots(figsize=(9.5, 5.8))
    for _, row in top1.iterrows():
        values = [float(row[column]) for column in metric_columns]
        category = str(row["defense_category"])
        color = DEFENSE_COLORS.get(category, "#4A5568")
        ax.plot(
            x_positions,
            values,
            marker="o",
            markersize=4.8,
            linewidth=1.8,
            color=color,
            alpha=0.72,
        )
        ax.text(
            x_positions[-1] + 0.05,
            values[-1],
            str(row["action_id"]),
            va="center",
            fontsize=TEXT_SIZE,
            color=color,
            fontproperties=zh_font,
        )

    ax.set_xlim(-0.08, len(metric_columns) - 0.70)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels, fontproperties=zh_font, fontsize=LABEL_SIZE - 1, rotation=15)
    ax.set_ylabel("归一化指标值", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 主防守 Top1 指标平行对比图", fontproperties=zh_font, fontsize=TITLE_SIZE)
    _set_axis_style(ax)
    _legend_outside(ax, figure, zh_font, include_na=False)
    return _save(figure, output_path)


def plot_decision_waterfall(matchup_table: pd.DataFrame, output_path: str | Path) -> Path:
    zh_font, _ = configure_fonts()
    table = matchup_table.copy().sort_values("active_top1_score", ascending=False).reset_index(drop=True)
    display_label = table["active_top1_defense_id"].astype(str).replace("", DISPLAY_EMPTY)

    figure, ax = plt.subplots(figsize=(11.0, 6.0))
    y = np.arange(len(table))
    bar_colors = [
        DEFENSE_COLORS.get("NA" if did == DISPLAY_EMPTY else category, "#285EAD")
        for did, category in zip(display_label, table["active_top1_category"].fillna("NA"), strict=True)
    ]
    ax.barh(y, np.maximum(table["active_top1_score"].astype(float), 0.0), color=bar_colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{aid}->{did}" for aid, did in zip(table["action_id"], display_label, strict=True)], fontsize=TICK_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontproperties(zh_font)
    ax.invert_yaxis()
    ax.set_xlabel("\u4e3b\u9632\u5b88\u8bc4\u5206", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_title("Q2 \u653b\u51fb\u52a8\u4f5c\u5230\u4e3b\u9632\u5b88 Top1 \u7684\u6620\u5c04\u56fe", fontproperties=zh_font, fontsize=TITLE_SIZE)
    _set_axis_style(ax)
    _legend_outside(ax, figure, zh_font, include_na=True)
    return _save(figure, output_path)


def plot_method_comparison(method_summary: pd.DataFrame, output_path: str | Path) -> Path:
    zh_font, _ = configure_fonts()
    table = method_summary.copy()
    attacks = table["action_id"].tolist()
    methods = ["method1", "method2", "method3", "method4"]
    figure, ax = plt.subplots(figsize=(11.0, 6.2))
    x_positions = np.arange(len(methods))

    for y_index, (_, row) in enumerate(table.iterrows()):
        for x_index, method_key in enumerate(methods):
            display_text, color_key = _method_cell_state(row, method_key)
            marker_color = NOT_APPLICABLE_COLOR if color_key == "not_applicable" else DEFENSE_COLORS.get(color_key, "#94A3B8")
            edge_color = "#CBD5E1" if color_key == "not_applicable" else "#111827"
            ax.scatter(
                x_positions[x_index],
                y_index,
                s=250,
                color=marker_color,
                edgecolors=edge_color,
                linewidths=0.45,
                zorder=3,
            )
            text_color = "#64748B" if color_key == "not_applicable" else "#1F2937" if color_key == "NA" else "white"
            ax.text(
                x_positions[x_index],
                y_index,
                display_text,
                ha="center",
                va="center",
                fontsize=TEXT_SIZE,
                color=text_color,
                weight="bold",
                fontproperties=zh_font,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[key] for key in methods], fontproperties=zh_font, fontsize=LABEL_SIZE - 1)
    ax.set_yticks(np.arange(len(attacks)))
    ax.set_yticklabels(attacks, fontsize=TICK_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontproperties(zh_font)
    ax.invert_yaxis()
    ax.set_title("Q2 \u56db\u79cd\u65b9\u6cd5 Top1 \u54cd\u5e94\u5bf9\u6bd4\u56fe", fontproperties=zh_font, fontsize=TITLE_SIZE)
    ax.set_xlabel("\u8bc4\u4f30\u65b9\u6cd5", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("\u653b\u51fb\u52a8\u4f5c", fontproperties=zh_font, fontsize=LABEL_SIZE)
    _set_axis_style(ax)
    _legend_outside(ax, figure, zh_font, include_na=True, include_not_applicable=True)
    return _save(figure, output_path)


def plot_layered_response_overview(method_summary: pd.DataFrame, output_path: str | Path) -> Path:
    zh_font, _ = configure_fonts()
    layers = [
        ("active", "method4_active_top1", "method4_active_category"),
        ("fallback", "method4_fallback_top1", "method4_fallback_category"),
        ("ground", "method4_ground_top1", "method4_ground_category"),
        ("recovery", "method4_recovery_if_needed", "method4_recovery_category"),
    ]

    figure, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-0.5, len(method_summary) - 0.5)

    for y_index, (_, row) in enumerate(method_summary.iterrows()):
        for x_index, (layer_key, column, category_column) in enumerate(layers):
            display_text, color_key, state_kind = _layer_cell_state(row, layer_key, column, category_column)
            if state_kind == "not_applicable":
                ax.text(
                    x_index,
                    y_index,
                    display_text,
                    ha="center",
                    va="center",
                    fontsize=TEXT_SIZE + 1,
                    color="#64748B",
                    fontproperties=zh_font,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": NOT_APPLICABLE_COLOR, "edgecolor": "none"},
                )
                continue

            ax.scatter(
                x_index,
                y_index,
                s=320,
                color=DEFENSE_COLORS.get(color_key, "#CBD5E1"),
                edgecolors="#111827",
                linewidths=0.45,
            )
            text_color = "#1F2937" if color_key == "NA" else "white"
            ax.text(
                x_index,
                y_index,
                display_text,
                ha="center",
                va="center",
                fontsize=TEXT_SIZE,
                color=text_color,
                weight="bold",
                fontproperties=zh_font,
            )

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([LAYER_LABELS[layer_key] for layer_key, _, _ in layers], fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_yticks(np.arange(len(method_summary)))
    ax.set_yticklabels(method_summary["action_id"].tolist(), fontsize=TICK_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontproperties(zh_font)
    ax.invert_yaxis()
    ax.set_title("Q2 \u5206\u5c42\u54cd\u5e94\u603b\u89c8\u56fe", fontproperties=zh_font, fontsize=TITLE_SIZE)
    ax.set_xlabel("\u54cd\u5e94\u5c42\u7ea7", fontproperties=zh_font, fontsize=LABEL_SIZE)
    ax.set_ylabel("\u653b\u51fb\u52a8\u4f5c", fontproperties=zh_font, fontsize=LABEL_SIZE)
    _set_axis_style(ax, show_grid=False)
    _legend_outside(ax, figure, zh_font, include_na=True, include_not_applicable=True)
    return _save(figure, output_path)
