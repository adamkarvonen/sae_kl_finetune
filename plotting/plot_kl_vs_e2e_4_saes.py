import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up directories
data_dir = "paper_data/gemma_LoRA_from_scratch_kl_vs_e2e_4_SAEs_data.csv"
image_dir = "paper_images"
os.makedirs(image_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(data_dir)

# Set up the figure with improved styling
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

# Extract the step column (first column)
step_col = df.columns[0]

# Define color scheme for k values
colors = {
    "20": "#1f77b4",  # blue
    "40": "#2ca02c",  # green
    "80": "#d62728",  # red
    "160": "#9467bd",  # purple
}

# Define line styles for KL types
line_styles = {
    "kl_0": "-",  # solid line for KL0
    "kl_95": "--",  # dashed line for KL95
}

# Create a dictionary to store dataframes for each line to avoid NaN issues
line_dfs = {}

# Extract relevant columns for plotting
for col in df.columns:
    if "MIN" in col or "MAX" in col:
        continue  # Skip min/max columns

    if col == step_col:
        continue  # Skip step column

    # Clean and prepare the data
    line_df = df[[step_col, col]].copy()
    line_df = line_df[line_df[col] != ""]
    line_df = line_df.dropna(subset=[col])
    line_df[col] = line_df[col].astype(float)

    line_dfs[col] = line_df

# Plot each line with appropriate styling
for col, line_df in line_dfs.items():
    # Extract k value and KL type from column name
    if "kl_0" in col.lower():
        kl_type = "kl_0"
        label_prefix = "E2E"
    else:
        kl_type = "kl_95"
        label_prefix = "MSE + KL Finetune"

    # Extract k value from column name
    for k in ["20", "40", "80", "160"]:
        if f"k_{k}" in col.lower():
            k_value = k
            break
    else:
        continue  # Skip if k value not found

    # Plot the line with appropriate styling
    plt.plot(
        line_df[step_col],
        line_df[col],
        linestyle=line_styles[kl_type],
        linewidth=2,
        color=colors[k_value],
        label=f"{label_prefix}, k={k_value}",
    )

# Add horizontal line for original model loss
original_model_loss = 2.0759266334277946
y_min = original_model_loss - 0.01
plt.axhline(
    y=original_model_loss,
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="Original Model Loss",
)

# Add labels and styling
plt.xlabel("Training Tokens", fontweight="bold", fontsize=14)
plt.ylabel("Validation Loss", fontweight="bold", fontsize=14)

# Set custom y-axis limit
plt.autoscale(tight=True)
plt.ylim(bottom=y_min, top=2.45)

# Add grid for better readability

plt.grid(True, linestyle="--", alpha=0.7)

# Add legend with better organization
plt.legend(
    frameon=True,
    fontsize=10,
    loc="upper right",
    title="Configuration",
    ncol=2,  # Organize in two columns for better space usage
)

# Ensure tight layout
plt.tight_layout()

# Save the figure
plt.savefig(
    f"{image_dir}/gemma_lora_multi_sae_comparison.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
