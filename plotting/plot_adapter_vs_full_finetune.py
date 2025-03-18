import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/adapters_vs_full_finetune.csv"  # Update path to your new dataset
# Load the CSV file
df = pd.read_csv(data_dir)
image_dir = "paper_images"
os.makedirs(image_dir, exist_ok=True)

# Set up the figure
plt.figure(figsize=(12, 7))  # Slightly larger figure for more lines
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

# Explicitly get column names to avoid indexing issues
all_cols = list(df.columns)
step_col = all_cols[0]  # "Step" column

# Define the columns to plot
rank_768_mlp_adapter_col = all_cols[
    1
]  # "layer_12_rank_768_mlp_adapter_only_v4 - val_loss"
rank_768_adapter_col = all_cols[4]  # "layer_12_rank_768_adapter_only_v4 - val_loss"
rank_16_mlp_adapter_col = all_cols[
    7
]  # "layer_12_rank_16_mlp_adapter_only_v4 - val_loss"
rank_16_adapter_col = all_cols[10]  # "layer_12_rank_16_adapter_only_v4 - val_loss"
sae_ft_col = all_cols[13]  # "layer_12_rank_16_sae_full_finetune_v4 - val_loss"

# Define more readable labels
labels = {
    rank_768_mlp_adapter_col: "Rank 768 MLP Adapter",
    rank_768_adapter_col: "Rank 768 Linear Transform",
    rank_16_mlp_adapter_col: "Rank 16 MLP Adapter",
    rank_16_adapter_col: "Rank 16 Linear Transform",
    sae_ft_col: "Full Fine Tune",
}

# Colors for each line
colors = {
    rank_768_mlp_adapter_col: "#1f77b4",  # blue
    rank_768_adapter_col: "#ff7f0e",  # orange
    rank_16_mlp_adapter_col: "#2ca02c",  # green
    rank_16_adapter_col: "#d62728",  # red
    sae_ft_col: "#9467bd",  # purple
}

# Process and plot each line
all_min_values = []
all_max_values = []
for col in [
    rank_768_mlp_adapter_col,
    rank_768_adapter_col,
    rank_16_mlp_adapter_col,
    rank_16_adapter_col,
    sae_ft_col,
]:
    # Filter and create separate DataFrame for each line to avoid NaN issues
    line_df = df[[step_col, col]].copy()
    line_df = line_df[line_df[col] != ""]
    line_df = line_df.dropna(subset=[col])
    line_df[col] = line_df[col].astype(float)

    # Convert steps to millions of tokens (each step is 1000 tokens)
    line_df[step_col] = line_df[step_col] / 1000

    # Store min value for y-axis scaling
    if not line_df.empty:
        all_min_values.append(line_df[col].min())
        all_max_values.append(line_df[col].max())
    # Plot the line
    plt.plot(
        line_df[step_col],
        line_df[col],
        linestyle="-",
        linewidth=2,
        color=colors[col],
        label=labels[col],
    )

# Add horizontal line for original model loss
original_model_loss = 1.923762257963022  # Adjust this if needed for your dataset
y_min = (
    min(all_min_values + [original_model_loss]) - 0.01
    if all_min_values
    else original_model_loss - 0.01
)
y_max = max(all_max_values + [original_model_loss]) + 0.01

plt.axhline(
    y=original_model_loss,
    color="k",  # black for better visibility
    linestyle="--",
    linewidth=1.5,
    label="Original Model Loss (best achievable)",
)

# Add labels
plt.xlabel("Training Tokens (millions)", fontweight="bold", fontsize=14)
plt.ylabel("Validation Loss", fontweight="bold", fontsize=14)

# Set custom y-axis limit
plt.autoscale(tight=True)
# Adjust top limit as needed
plt.ylim(bottom=y_min, top=y_max)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend with smaller font to accommodate more entries
plt.legend(frameon=True, fontsize=10, loc="best")

# Ensure tight layout
plt.tight_layout()

# Save the figure with the new filename
plt.savefig(
    f"{image_dir}/adaptation_methods_comparison.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)

# plt.show()
