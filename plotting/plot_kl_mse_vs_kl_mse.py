import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/kl_mse_vs_kl_mse.csv"
# Load the CSV file
df = pd.read_csv(data_dir)
image_dir = "paper_images"
os.makedirs(image_dir, exist_ok=True)


def smooth_data(y, alpha=0.1):
    """Apply exponential moving average smoothing"""
    smoothed = []
    last = y[0]
    for point in y:
        smoothed_val = alpha * point + (1 - alpha) * last
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Set up the figure
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

# Explicitly get column names to avoid indexing issues
all_cols = list(df.columns)
step_col = all_cols[0]  # "Step" column
e2e_col = all_cols[1]  # KL0 column
msc_col = all_cols[4]  # KL95 column

# Filter and create separate DataFrames for each line to avoid NaN issues
e2e_df = df[[step_col, e2e_col]].copy()
e2e_df = e2e_df[e2e_df[e2e_col] != ""]
e2e_df = e2e_df.dropna(subset=[e2e_col])
e2e_df[e2e_col] = e2e_df[e2e_col].astype(float)

msc_df = df[[step_col, msc_col]].copy()
msc_df = msc_df[msc_df[msc_col] != ""]
msc_df = msc_df.dropna(subset=[msc_col])
msc_df[msc_col] = msc_df[msc_col].astype(float)

# Calculate smoothed values
e2e_smoothed = smooth_data(e2e_df[e2e_col].values)
msc_smoothed = smooth_data(msc_df[msc_col].values)

# Calculate the maximum y-value from both datasets
max_y = max(max(e2e_smoothed), max(msc_smoothed))
y_max = max_y + 0.03
min_y = min(min(e2e_smoothed), min(msc_smoothed))
y_min = min_y - 0.03

# Plot original data with low opacity
plt.plot(
    e2e_df[step_col],
    e2e_df[e2e_col],
    linestyle="-",
    linewidth=1,
    color="#1f77b4",
    alpha=0.2,
)

plt.plot(
    msc_df[step_col],
    msc_df[msc_col],
    linestyle="-",
    linewidth=1,
    color="#ff7f0e",
    alpha=0.2,
)

# Plot smoothed data
plt.plot(
    e2e_df[step_col],
    e2e_smoothed,
    linestyle="-",
    linewidth=2.5,
    color="#1f77b4",
    label="KL only",
)

plt.plot(
    msc_df[step_col],
    msc_smoothed,
    linestyle="--",
    linewidth=2.5,
    color="#ff7f0e",
    label="MSE + KL",
)

# Add labels
plt.xlabel("Training Tokens", fontweight="bold", fontsize=14)
plt.ylabel("Training MSE (with EMA smoothing)", fontweight="bold", fontsize=14)
# plt.title("Gemma LoRA Training Comparison", fontsize=16)

# Set custom y-axis limit
plt.autoscale(tight=True)
# plt.ylim(bottom=y_min, top=y_max)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend(frameon=True, fontsize=12)

# Ensure tight layout
plt.tight_layout()

# Save the figure
plt.savefig(
    f"{image_dir}/kl_mse_vs_kl_mse.png", format="png", dpi=300, bbox_inches="tight"
)

# plt.show()
