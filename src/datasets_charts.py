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
    """Convert subgroup to string, removing .0 from integers"""
    subgroup_str = str(subgroup)
    # If it ends with .0, remove it (e.g., "20.0" -> "20")
    if subgroup_str.endswith('.0'):
        subgroup_str = subgroup_str[:-2]
    return subgroup_str

# Colors from your RGB table
color_map = {
    ("Control", "0.5"): rgb_to_hex(0, 0, 0),           # Black
    ("Control", "1.0"): rgb_to_hex(0, 0, 0),           # Black
    ("CytochalasinD", "20"): rgb_to_hex(14, 11, 124),  # #0e0b7c
    ("CytochalasinD", "50"): rgb_to_hex(28, 197, 254), # #1cc5fe
    ("Taxol", "100"): rgb_to_hex(164, 138, 211),       # #a48ad3
    ("Taxol", "200"): rgb_to_hex(164, 138, 211),       # #a48ad3
    ("Nocodazole", "10"): rgb_to_hex(251, 0, 5),       # #fb0005
    ("Nocodazole", "1"): rgb_to_hex(251, 79, 6),       # #fb4f06
}

# Get unique group-subgroup combinations
groups = timeline_df.groupby(["group", "subgroup"]).first().reset_index()

# ============================================================================
# BOXPLOTS
# ============================================================================

# Create boxplots for displacement, area, and max_feret_diameter
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = [
    ("mean_displacement_um", "Displacement (μm)"),
    ("mean_area_um2", "Area (μm²)"),
    ("mean_max_feret_diameter_um", "Max Feret Diameter (μm)")
]

for ax, (metric, ylabel) in zip(axes, metrics):
    boxplot_data = []
    labels = []
    colors = []
    
    for _, row in groups.iterrows():
        group = row["group"]
        subgroup = normalize_subgroup(row["subgroup"])
        
        # Filter data
        data = boxplot_df[
            (boxplot_df["group"] == group) & 
            (boxplot_df["subgroup"].apply(normalize_subgroup) == subgroup)
        ][metric].values
        
        boxplot_data.append(data)
        labels.append(f"{group}\n{subgroup}")
        color = color_map.get((group, subgroup), "#808080")
        colors.append(color)
        print(f"Group: {group}, Subgroup: {subgroup}, Color: {color}")  # Debug print
    
    bp = ax.boxplot(boxplot_data, tick_labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} by Group", fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("boxplots_all_metrics.png", dpi=150)
plt.close()

print("Saved: boxplots_all_metrics.png")

# ============================================================================
# TIMELINE CHARTS WITH STANDARD ERROR
# ============================================================================

metrics_timeline = [
    ("mean_displacement_um", "se_displacement_um", "Displacement (μm)"),
    ("mean_area_um2", "se_area_um2", "Area (μm²)"),
    ("mean_max_feret_diameter_um", "se_max_feret_diameter_um", "Max Feret Diameter (μm)")
]

for metric, se_metric, ylabel in metrics_timeline:
    plt.figure(figsize=(16, 6))
    
    for _, row in groups.iterrows():
        group = row["group"]
        subgroup = normalize_subgroup(row["subgroup"])
        color = color_map.get((group, subgroup), "#808080")
        
        group_data = timeline_df[
            (timeline_df["group"] == group) & 
            (timeline_df["subgroup"].apply(normalize_subgroup) == subgroup)
        ].sort_values("frame")
        
        frames = group_data["frame"].values
        means = group_data[metric].values
        se = group_data[se_metric].values
        
        label = f"{group} - {subgroup}"
        
        plt.plot(frames, means, linewidth=2, label=label, color=color)
        plt.fill_between(
            frames,
            means - se,
            means + se,
            alpha=0.2,
            color=color
        )
    
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} over Frames - All Groups", fontsize=14)
    
    # Set x-axis ticks
    max_frame = int(timeline_df["frame"].max())
    plt.xticks(range(0, max_frame + 1, 1000))
    
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    # Create filename from metric name
    filename = metric.replace("mean_", "").replace("_um", "").replace("_um2", "")
    plt.savefig(f"timeline_{filename}.png", dpi=150)
    plt.close()
    
    print(f"Saved: timeline_{filename}.png")

print("Done!")
