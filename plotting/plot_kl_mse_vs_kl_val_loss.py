import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/kl_mse_vs_kl_val_loss.csv"
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

# Calculate the maximum y-value from both datasets
max_y = max(e2e_df[e2e_col].max(), msc_df[msc_col].max())
y_max = max_y + 0.03
min_y = min(e2e_df[e2e_col].min(), msc_df[msc_col].min())
y_min = min_y - 0.03

# Plot each line separately using clean DataFrames
plt.plot(
    e2e_df[step_col],
    e2e_df[e2e_col],
    linestyle="-",
    linewidth=2.5,
    color="#1f77b4",
    label="KL only",
)

plt.plot(
    msc_df[step_col],
    msc_df[msc_col],
    linestyle="--",
    linewidth=2.5,
    color="#ff7f0e",
    label="MSE + KL",
)

# Add horizontal line for original model loss
# original_model_loss = 1.9574154930812675
# y_min = original_model_loss - 0.01
# plt.axhline(
#     y=original_model_loss,
#     color="r",
#     linestyle=":",
#     linewidth=1.5,
#     label="Original Model Loss (best achievable)",
# )

# Add labels
plt.xlabel("Training Tokens", fontweight="bold", fontsize=14)
plt.ylabel("Validation Loss", fontweight="bold", fontsize=14)
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
    f"{image_dir}/kl_mse_vs_kl_val_loss.png", format="png", dpi=300, bbox_inches="tight"
)

# plt.show()
