import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/gemma_LoRA_from_scratch_kl_vs_e2e_fig1_data.csv"
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
# Convert steps to millions of tokens (each step is 1000 tokens)
e2e_df[step_col] = e2e_df[step_col] / 1000

msc_df = df[[step_col, msc_col]].copy()
msc_df = msc_df[msc_df[msc_col] != ""]
msc_df = msc_df.dropna(subset=[msc_col])
msc_df[msc_col] = msc_df[msc_col].astype(float)
# Convert steps to millions of tokens (each step is 1000 tokens)
msc_df[step_col] = msc_df[step_col] / 1000

# Plot each line separately using clean DataFrames
plt.plot(
    e2e_df[step_col],
    e2e_df[e2e_col],
    linestyle="-",
    linewidth=2,
    color="#1f77b4",
    label="E2E",
)

plt.plot(
    msc_df[step_col],
    msc_df[msc_col],
    linestyle="-",
    linewidth=2,
    color="#ff7f0e",
    label="MSE + KL fine tune",
)

# Add horizontal line for original model loss
original_model_loss = 2.0759266334277946
y_min = original_model_loss - 0.01
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
# plt.title("Gemma LoRA Training Comparison", fontsize=16)

# Set custom y-axis limit
plt.autoscale(tight=True)
plt.ylim(bottom=y_min, top=2.25)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend(frameon=True, fontsize=12)

# Ensure tight layout
plt.tight_layout()

# Save the figure
plt.savefig(
    f"{image_dir}/gemma_lora_comparison.png", format="png", dpi=300, bbox_inches="tight"
)

# plt.show()
