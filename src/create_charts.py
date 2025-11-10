import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
boxplot_df = pd.read_csv("data_for_boxplots.csv")
timeline_df = pd.read_csv("data_for_timeline_charts.csv")


# RGB to hex conversion
def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


# Function to normalize subgroup to string format matching the color_map keys
def normalize_subgroup(subgroup):
    """Convert subgroup to string, keeping 1.0 as 1.0 and removing .0 from other integers"""
    subgroup_str = str(subgroup)
    # Convert "1" to "1.0" for consistency
    if subgroup_str == "1":
        return "1.0"
    # If it ends with .0, remove it (e.g., "20.0" -> "20") but keep "1.0" as "1.0"
    if subgroup_str.endswith(".0") and subgroup_str != "1.0":
        subgroup_str = subgroup_str[:-2]
    return subgroup_str


# Colors and opacity from your updated table
color_map = {
    ("Control", "0.5"): rgb_to_hex(0, 0, 0),  # Black
    ("Control", "1.0"): rgb_to_hex(120, 120, 120),  # Medium gray
    ("CytochalasinD", "20"): rgb_to_hex(14, 111, 124),  # Muted teal/cyan
    ("CytochalasinD", "50"): rgb_to_hex(100, 200, 230),  # Soft light blue
    ("Taxol", "100"): rgb_to_hex(100, 80, 150),  # Muted purple
    ("Taxol", "200"): rgb_to_hex(200, 150, 220),  # Light lavender/pink
    ("Nocodazole", "10"): rgb_to_hex(240, 100, 100),  # Soft red
    ("Nocodazole", "1.0"): rgb_to_hex(250, 150, 80),  # Soft orange
}

# Border opacity (for lines and box edges) - converted to 0-1 scale
border_opacity_map = {
    ("Control", "0.5"): 100 / 100,
    ("Control", "1.0"): 100 / 100,
    ("CytochalasinD", "20"): 100 / 100,
    ("CytochalasinD", "50"): 100 / 100,
    ("Taxol", "100"): 100 / 100,
    ("Taxol", "200"): 100 / 100,
    ("Nocodazole", "10"): 100 / 100,
    ("Nocodazole", "1.0"): 100 / 100,
}

# Core opacity (for filled areas) - converted to 0-1 scale
core_opacity_map = {
    ("Control", "0.5"): 10 / 100,
    ("Control", "1.0"): 15 / 100,  # Very light to see black line through
    ("CytochalasinD", "20"): 20 / 100,  # Light to see lines better
    ("CytochalasinD", "50"): 20 / 100,  # Light to see lines better
    ("Taxol", "100"): 25 / 100,  # Light to reduce overlap
    ("Taxol", "200"): 25 / 100,  # Light to reduce overlap
    ("Nocodazole", "10"): 25 / 100,  # Light to reduce overlap
    ("Nocodazole", "1.0"): 25 / 100,  # Light to reduce overlap
}

# Unit mapping for subgroups
unit_map = {
    ("Control", "0.5"): "%",
    ("Control", "1.0"): "%",
    ("CytochalasinD", "20"): "μM",
    ("CytochalasinD", "50"): "μM",
    ("Taxol", "100"): "nM",
    ("Taxol", "200"): "nM",
    ("Nocodazole", "10"): "μM",
    ("Nocodazole", "1.0"): "μM",
}

# Get unique group-subgroup combinations
groups = timeline_df.groupby(["group", "subgroup"]).first().reset_index()

print("Creating charts...")

# ============================================================================
# BOXPLOTS - SEPARATE CHART FOR EACH METRIC
# ============================================================================

metrics = [
    ("mean_displacement_um", "Displacement (μm) -- All groups"),
    ("mean_area_um2", "Area (μm²) -- All groups"),
    ("mean_max_feret_diameter_um", "Max Feret Diameter (μm) -- All groups"),
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
        subgroup = normalize_subgroup(row["subgroup"])

        # Filter data
        data = boxplot_df[
            (boxplot_df["group"] == group)
            & (boxplot_df["subgroup"].apply(normalize_subgroup) == subgroup)
        ][metric].values

        boxplot_data.append(data)
        labels.append(f"{group}\n{subgroup}")
        color = color_map.get((group, subgroup), "#808080")
        colors.append(color)
        core_opacities.append(core_opacity_map.get((group, subgroup), 0.6))
        border_opacities.append(border_opacity_map.get((group, subgroup), 1.0))

    bp = ax.boxplot(boxplot_data, tick_labels=labels, patch_artist=True, widths=0.6)

    # Color the boxes with core opacity for fill and border opacity for edges
    for patch, color, core_alpha, border_alpha in zip(
        bp["boxes"], colors, core_opacities, border_opacities
    ):
        patch.set_facecolor(color)
        patch.set_alpha(core_alpha)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        # Apply border opacity to edge
        edge_color = patch.get_edgecolor()
        patch.set_edgecolor((*edge_color[:3], border_alpha))

    # Style whiskers, caps, and fliers with border opacity
    for i, border_alpha in enumerate(border_opacities):
        color = colors[i]
        # Convert hex to RGB
        r, g, b = int(color[1:3], 16) / 255, int(color[3:5], 16) / 255, int(color[5:7], 16) / 255
        
        # Whiskers
        bp["whiskers"][i * 2].set_color((r, g, b, border_alpha))
        bp["whiskers"][i * 2].set_linewidth(1.5)
        bp["whiskers"][i * 2 + 1].set_color((r, g, b, border_alpha))
        bp["whiskers"][i * 2 + 1].set_linewidth(1.5)
        
        # Caps
        bp["caps"][i * 2].set_color((r, g, b, border_alpha))
        bp["caps"][i * 2].set_linewidth(1.5)
        bp["caps"][i * 2 + 1].set_color((r, g, b, border_alpha))
        bp["caps"][i * 2 + 1].set_linewidth(1.5)
        
        # Fliers (outliers)
        bp["fliers"][i].set_markeredgecolor((r, g, b, border_alpha))
        bp["fliers"][i].set_markerfacecolor((r, g, b, core_opacities[i]))

    # Make median lines black and more visible
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Add individual data points with core opacity
    for i, (data, color, core_alpha) in enumerate(zip(boxplot_data, colors, core_opacities), 1):
        # Add jitter to x-coordinates so points don't overlap
        x = np.random.normal(i, 0.04, size=len(data))
        r, g, b = int(color[1:3], 16) / 255, int(color[3:5], 16) / 255, int(color[5:7], 16) / 255
        ax.scatter(x, data, alpha=core_alpha, s=20, color=(r, g, b), zorder=3, edgecolors='none')

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} by Group", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Create filename from metric name
    filename = metric.replace("mean_", "").replace("_um", "").replace("_um2", "")
    plt.savefig(f"boxplot_{filename}.png", dpi=150)
    plt.close()

    print(f"Saved: boxplot_{filename}.png")

# ============================================================================
# TIMELINE CHARTS WITH STANDARD ERROR - SEPARATE CHART FOR EACH METRIC
# ============================================================================

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
        subgroup = normalize_subgroup(row["subgroup"])
        color = color_map.get((group, subgroup), "#808080")
        border_alpha = border_opacity_map.get((group, subgroup), 1.0)
        core_alpha = core_opacity_map.get((group, subgroup), 0.2)

        group_data = timeline_df[
            (timeline_df["group"] == group)
            & (timeline_df["subgroup"].apply(normalize_subgroup) == subgroup)
        ].sort_values("frame")

        frames = group_data["frame"].values
        means = group_data[metric].values
        se = group_data[se_metric].values

        # Get the unit for this group/subgroup
        unit = unit_map.get((group, subgroup), "")
        label = f"{subgroup} {unit} {group}"

        # Plot line with border opacity
        plt.plot(frames, means, linewidth=3.0, label=label, color=color, alpha=border_alpha)
        
        # Fill area with core opacity
        plt.fill_between(frames, means - se, means + se, alpha=core_alpha, color=color)

    plt.xlabel("Frame", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel}", fontsize=14)

    # Set x-axis range to match the data exactly
    min_frame = int(timeline_df["frame"].min())
    max_frame = int(timeline_df["frame"].max())
    plt.xlim(min_frame, max_frame)
    plt.xticks(range(0, max_frame + 1, 1000))

    plt.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Create filename from metric name
    filename = metric.replace("mean_", "").replace("_um", "").replace("_um2", "")
    plt.savefig(f"timeline_{filename}.png", dpi=150)
    plt.close()

    print(f"Saved: timeline_{filename}.png")

print("\nAll charts created successfully!")
print("\nBoxplots:")
print("  - boxplot_displacement.png")
print("  - boxplot_area.png")
print("  - boxplot_max_feret_diameter.png")
print("\nTimeline charts:")
print("  - timeline_displacement.png")
print("  - timeline_area.png")
print("  - timeline_max_feret_diameter.png")
