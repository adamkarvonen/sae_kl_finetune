import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/adapter_two_stage.csv"  # Update path to your dataset
# Load the CSV file
df = pd.read_csv(data_dir)
image_dir = "paper_images"
os.makedirs(image_dir, exist_ok=True)

# Set up the figure
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

# Explicitly get column names to avoid indexing issues
all_cols = list(df.columns)
step_col = all_cols[0]  # "Step" column
rank_768_col = all_cols[1]  # "layer_12_rank_768_adapter_and_sae_v4 - val_loss"
rank_16_col = all_cols[4]  # "layer_12_rank_16_adapter_and_sae_v4 - val_loss"

# Clean data and convert to float
df[rank_768_col] = pd.to_numeric(df[rank_768_col], errors="coerce")
df[rank_16_col] = pd.to_numeric(df[rank_16_col], errors="coerce")
df = df.dropna(subset=[rank_768_col, rank_16_col])

# Convert steps to millions of tokens (each step is 1000 tokens)
df[step_col] = df[step_col] / 1000

# The actual transition step is between 14998 and 15998 (based on data)
# Using 15500 as a visual transition point - halfway between the actual data points
transition_step = 15.5  # Converted to millions of tokens

# Create separate dataframes but include the transition points in both
# to ensure there's no gap in the plot lines
df_before = df[df[step_col] <= 16.0].copy()  # Include first point after transition
df_after = df[df[step_col] >= 15.0].copy()  # Include last point before transition

# Plot the first stage (SAE-only) with dashed lines
plt.plot(
    df_before[step_col],
    df_before[rank_768_col],
    linestyle="--",
    linewidth=2,
    color="#1f77b4",  # blue
    label="Rank 768 (SAE training)",
)

plt.plot(
    df_before[step_col],
    df_before[rank_16_col],
    linestyle="--",
    linewidth=2,
    color="#ff7f0e",  # orange
    label="Rank 16 (SAE training)",
)

# Plot the second stage (Adapter-only) with solid lines
plt.plot(
    df_after[step_col],
    df_after[rank_768_col],
    linestyle="-",
    linewidth=2,
    color="#1f77b4",  # blue
    label="Rank 768 (Adapter training)",
)

plt.plot(
    df_after[step_col],
    df_after[rank_16_col],
    linestyle="-",
    linewidth=2,
    color="#ff7f0e",  # orange
    label="Rank 16 (Adapter training)",
)

# Add vertical line to indicate transition
plt.axvline(
    x=transition_step,
    color="k",
    linestyle="-.",
    linewidth=1.5,
    alpha=0.7,
    label="Training transition",
)

# Add text annotation for the stages
plt.text(
    transition_step / 2,
    df[rank_768_col].min() - 0.01,
    "SAE Training",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
)
plt.text(
    transition_step + (df[step_col].max() - transition_step) / 2,
    df[rank_768_col].min() - 0.01,
    "Adapter Training",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
)

# Add horizontal line for original model loss
original_model_loss = 1.923762257963022  # Adjusted value
y_min = min(df[rank_768_col].min(), df[rank_16_col].min(), original_model_loss) - 0.01
y_max = (
    max(df[rank_768_col].max(), df[rank_16_col].max()) + 0.03
)  # Give more space for labels

plt.axhline(
    y=original_model_loss,
    color="r",
    linestyle="--",
    linewidth=1.5,
    label="Original Model Loss (best achievable)",
)

# Add labels
plt.xlabel("Training Tokens (millions)", fontweight="bold", fontsize=14)
plt.ylabel("Validation Loss", fontweight="bold", fontsize=14)

# Set custom y-axis limit
plt.autoscale(tight=True)
plt.ylim(bottom=y_min, top=y_max)  # Adjusted limits

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend - place it in the upper right to avoid overlap with plot lines
plt.legend(frameon=True, fontsize=10, loc="upper right")

# Ensure tight layout
plt.tight_layout()

# Save the figure
plt.savefig(
    f"{image_dir}/two_stage_training.png", format="png", dpi=300, bbox_inches="tight"
)

# plt.show()
