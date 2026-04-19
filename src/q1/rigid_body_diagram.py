from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import pandas as pd

# --------------------------
# 仅前视图布局（文字位置彻底优化，无任何重叠）
# --------------------------
FRONT_LAYOUT = {
    "S1": (0.00, 0.90),
    "S2": (0.00, 0.60),
    "S3": (-0.22, 0.66),
    "S4": (0.22, 0.66),
    "S5": (-0.42, 0.55),
    "S6": (0.42, 0.55),
    "S7": (-0.12, 0.32),
    "S8": (0.12, 0.32),
    "S9": (-0.12, 0.07),
    "S10": (0.12, 0.07),
}

VIEW_TITLES = {
    "front": "Q1 十刚体前视示意图",
}

# --------------------------
# 动态路径配置（根据你的目录结构修正）
# --------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]  # 上两级到项目根目录
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "q1_segment_params.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

def configure_fonts() -> None:
    preferred_cn = ["SimSun", "宋体"]
    preferred_en = ["Times New Roman", "Times New Roman PS MT"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    cn_font = next((name for name in preferred_cn if name in available), "DejaVu Sans")
    en_font = next((name for name in preferred_en if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [cn_font, en_font]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

def load_segments() -> pd.DataFrame:
    data = pd.read_csv(RAW_FILE)
    return data.sort_values("segment_id").reset_index(drop=True)

def draw_view(data: pd.DataFrame, view_name: str, layout: dict[str, tuple[float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 8.0), dpi=520)
    colors = {
        "center": "#324b3a",
        "left": "#9b6a32",
        "right": "#35618f",
    }

    for _, row in data.iterrows():
        segment_id = str(row["segment_id"])
        parent_id = "" if pd.isna(row["parent_id"]) else str(row["parent_id"])
        x, y = layout[segment_id]
        color = colors.get(str(row["side"]), "#555555")

        # 关节连接线
        if parent_id and parent_id in layout:
            px, py = layout[parent_id]
            ax.plot([px, x], [py, y], color="#555555", linewidth=2.0, zorder=1)

        # 刚体圆
        radius = 0.045
        circle = Circle((x, y), radius=radius, facecolor=color, edgecolor="white", linewidth=1.8, zorder=2)
        ax.add_patch(circle)

        # 节段质心白点
        com_y = y - radius * 0.22
        ax.scatter([x], [com_y], s=20, c="white", edgecolors="#222222", linewidths=0.5, zorder=3)

        # ==================== 终极文字排版（完全无重叠） ====================
        # 1. 头部（S1）
        if segment_id == "S1":
            ax.text(x, y + radius + 0.04, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            # 名称放在圆下方，远离S2
            ax.text(x, y - radius - 0.02, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="center", va="top", fontsize=9)

        # 2. 躯干（S2）
        elif segment_id == "S2":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            # 名称放在圆的正下方，不与其他任何文字重叠
            ax.text(x, y - radius - 0.05, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="center", va="top", fontsize=9)

        # 3. 左上臂、左前臂（S3、S5）
        elif segment_id == "S3":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            # 名称向圆的左侧下方偏移，远离躯干
            ax.text(x - 0.08, y - radius - 0.01, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="right", va="top", fontsize=9)
        elif segment_id == "S5":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.text(x - 0.08, y - radius - 0.01, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="right", va="top", fontsize=9)

        # 4. 右上臂、右前臂（S4、S6）
        elif segment_id == "S4":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            # 名称向圆的右侧下方偏移，远离躯干
            ax.text(x + 0.08, y - radius - 0.01, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="left", va="top", fontsize=9)
        elif segment_id == "S6":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.text(x + 0.08, y - radius - 0.01, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="left", va="top", fontsize=9)

        # 5. 左大腿、左小腿（S7、S9）
        elif segment_id == "S7":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            # 名称向圆的左侧下方偏移
            ax.text(x - 0.06, y - radius - 0.01, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="right", va="top", fontsize=9)
        elif segment_id == "S9":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.text(x, y - radius - 0.05, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="center", va="top", fontsize=9)

        # 6. 右大腿、右小腿（S8、S10）
        elif segment_id == "S8":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            # 名称向圆的右侧下方偏移
            ax.text(x + 0.06, y - radius - 0.01, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="left", va="top", fontsize=9)
        elif segment_id == "S10":
            ax.text(x, y + radius + 0.02, segment_id, ha="center", va="bottom", fontsize=12, fontweight="bold")
            ax.text(x, y - radius - 0.05, f"{row['segment_name_cn']}\n{row['segment_name_en']}",
                    ha="center", va="top", fontsize=9)

    ax.set_title(VIEW_TITLES[view_name], fontsize=16, pad=12)
    ax.text(
        0.02, 0.98, "白点：节段质心\n灰线：关节连接关系",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"}
    )

    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.05, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"q1_rigid_body_{view_name}.png", dpi=520, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    configure_fonts()
    data = load_segments()
    draw_view(data, "front", FRONT_LAYOUT)
    print("✅ 前视图 PNG 生成完成！所有文字无重叠 ✅")

if __name__ == "__main__":
    main()