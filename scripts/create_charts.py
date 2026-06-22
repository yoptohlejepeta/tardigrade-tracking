"""Create boxplot and line charts.

Line charts and boxplots for area, displacement and max feret diameter.

- Boxplot: mean value by objects accross all frames
- Line chart: mean value for each frame by whole group+subgroup (e.g. Control 0.5)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

boxplot_df = pd.read_csv("data_for_boxplots.csv", dtype={"subgroup": str})
timeline_df = pd.read_csv("data_for_timeline_charts.csv", dtype={"subgroup": str})


color_map = {
    ("Control", "0.5"): "#000000",
    ("Control", "1.0"): "#787878",
    ("CytochalasinD", "20"): "#0E6F7C",
    ("CytochalasinD", "50"): "#64C8E6",
    # ("Taxol", "100"): "#645096",
    ("Taxol", "10"): "#645096",
    ("Taxol", "200"): "#C896DC",
    ("Nocodazole", "10"): "#F06464",
    ("Nocodazole", "1"): "#FA9650",
    ("Jasplakinolide", "1"): "#38A848",
    ("Jasplakinolide", "10"): "#8CDB96",
}

core_opacity_map = {
    ("Control", "0.5"): 10 / 100,
    ("Control", "1.0"): 15 / 100,
    ("CytochalasinD", "20"): 20 / 100,
    ("CytochalasinD", "50"): 20 / 100,
    ("Taxol", "10"): 25 / 100,
    ("Taxol", "200"): 25 / 100,
    ("Nocodazole", "10"): 25 / 100,
    ("Nocodazole", "1.0"): 25 / 100,
    ("Jasplakinolide", "1"): 20 / 100,
    ("Jasplakinolide", "10"): 20 / 100,
}

unit_map = {
    ("Control", "0.5"): "%",
    ("Control", "1.0"): "%",
    ("CytochalasinD", "20"): "μM",
    ("CytochalasinD", "50"): "μM",
    ("Taxol", "10"): "μM",
    ("Taxol", "200"): "nM",
    ("Nocodazole", "10"): "μM",
    ("Nocodazole", "1"): "μM",
    ("Jasplakinolide", "1"): "μM",
    ("Jasplakinolide", "10"): "μM",
}

groups = timeline_df.groupby(["group", "subgroup"]).first().reset_index()

print("Creating charts...")

metrics = [
    ("mean_displacement_um", "Displacement (μm) — All groups"),
    ("mean_area_um2", "Area (μm²) — All groups"),
    ("mean_max_feret_diameter_um", "Max Feret Diameter (μm) — All groups"),
]

for metric, ylabel in metrics:
    fig, ax = plt.subplots(figsize=(12, 6))

    boxplot_data = []
    labels = []
    colors = []
    core_opacities = []
    border_opacities = []

    for _, row in groups.iterrows():
        group = row["group"]
        subgroup = row["subgroup"]

        data = boxplot_df[
            (boxplot_df["group"] == group) & (boxplot_df["subgroup"] == subgroup)
        ][metric].values

        boxplot_data.append(data)
        labels.append(f"{group}\n{subgroup}")

        color = color_map.get((group, subgroup))
        colors.append(color)
        core_opacities.append(core_opacity_map.get((group, subgroup), 0.6))
        border_opacities.append(1)

    bp = ax.boxplot(boxplot_data, tick_labels=labels, patch_artist=True, widths=0.6)

    for patch, color, core_alpha, border_alpha in zip(
        bp["boxes"], colors, core_opacities, border_opacities
    ):
        patch.set_facecolor(color)
        patch.set_alpha(core_alpha)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        edge_color = patch.get_edgecolor()
        patch.set_edgecolor((*edge_color[:3], border_alpha))

    for i, border_alpha in enumerate(border_opacities):
        color = colors[i]
        r, g, b = (
            int(color[1:3], 16) / 255,
            int(color[3:5], 16) / 255,
            int(color[5:7], 16) / 255,
        )

        bp["whiskers"][i * 2].set_color((r, g, b, border_alpha))
        bp["whiskers"][i * 2].set_linewidth(1.5)
        bp["whiskers"][i * 2 + 1].set_color((r, g, b, border_alpha))
        bp["whiskers"][i * 2 + 1].set_linewidth(1.5)

        bp["caps"][i * 2].set_color((r, g, b, border_alpha))
        bp["caps"][i * 2].set_linewidth(1.5)
        bp["caps"][i * 2 + 1].set_color((r, g, b, border_alpha))
        bp["caps"][i * 2 + 1].set_linewidth(1.5)

        bp["fliers"][i].set_markeredgecolor((r, g, b, border_alpha))
        bp["fliers"][i].set_markerfacecolor((r, g, b, core_opacities[i]))

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    for i, (data, color, core_alpha) in enumerate(
        zip(boxplot_data, colors, core_opacities), 1
    ):
        x = np.random.normal(i, 0.04, size=len(data))
        r, g, b = (
            int(color[1:3], 16) / 255,
            int(color[3:5], 16) / 255,
            int(color[5:7], 16) / 255,
        )
        ax.scatter(
            x,
            data,
            alpha=core_alpha,
            s=20,
            color=(r, g, b),
            zorder=3,
            edgecolors="none",
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} by Group", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    filename = metric.replace("mean_", "").replace("_um", "").replace("_um2", "")
    plt.savefig(f"boxplot_{filename}.png", dpi=150)
    plt.close()

    print(f"Saved: boxplot_{filename}.png")

metrics_timeline = [
    (
        "mean_displacement_um",
        "se_displacement_um",
        "Tun Displacement (μm) — All groups",
    ),
    ("mean_area_um2", "se_area_um2", "Tun area (μm²) — All groups"),
    (
        "mean_max_feret_diameter_um",
        "se_max_feret_diameter_um",
        "Tun Max Feret Diameter (μm) — All groups",
    ),
]

for metric, se_metric, ylabel in metrics_timeline:
    plt.figure(figsize=(14, 6))

    for _, row in groups.iterrows():
        group = row["group"]
        subgroup = row["subgroup"]
        color = color_map.get((group, subgroup), "#808080")
        border_alpha = 1
        core_alpha = core_opacity_map.get((group, subgroup), 0.2)

        group_data = timeline_df[
            (timeline_df["group"] == group) & (timeline_df["subgroup"] == subgroup)
        ].sort_values("time_seconds")

        time = group_data["time_seconds"].values
        means = group_data[metric].values
        se = group_data[se_metric].values

        unit = unit_map.get((group, subgroup), "")
        label = f"{subgroup} {unit} {group}"

        # line with border opacity
        plt.plot(
            time, means, linewidth=3.0, label=label, color=color, alpha=border_alpha
        )

        # fill area with core opacity
        plt.fill_between(time, means - se, means + se, alpha=core_alpha, color=color)

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel}", fontsize=14)

    # min_time = int(timeline_df["time_seconds"].min())
    # max_time = int(timeline_df["time_seconds"].max())
    plt.xlim(0, 60)
    plt.xticks(range(0, 61, 15))

    plt.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    filename = metric.replace("mean_", "").replace("_um", "").replace("_um2", "")
    plt.savefig(f"timeline_{filename}.png", dpi=150)
    plt.close()

    print(f"Saved: timeline_{filename}.png")

print("\nAll charts created successfully!")
