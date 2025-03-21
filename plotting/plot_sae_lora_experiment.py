import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/sae_lora_experiment.csv"
# Load the CSV file
df = pd.read_csv(data_dir)
image_dir = "paper_images"
os.makedirs(image_dir, exist_ok=True)

# Set up the figure
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 18

# Explicitly get column names to avoid indexing issues
all_cols = list(df.columns)
step_col = all_cols[0]  # "Step" column
sae_lora_64_col = all_cols[1]  # "layer_12_rank_64_sae_lora - val_loss"
sae_lora_2_col = all_cols[4]  # "layer_12_rank_2_sae_lora - val_loss"
sae_ft_col = all_cols[7]  # "layer_12_rank_16_sae_full_finetune - val_loss"

# Filter and create separate DataFrames for each line to avoid NaN issues
sae_lora_64_df = df[[step_col, sae_lora_64_col]].copy()
sae_lora_64_df = sae_lora_64_df[sae_lora_64_df[sae_lora_64_col] != ""]
sae_lora_64_df = sae_lora_64_df.dropna(subset=[sae_lora_64_col])
sae_lora_64_df[sae_lora_64_col] = sae_lora_64_df[sae_lora_64_col].astype(float)
sae_lora_64_df[step_col] = sae_lora_64_df[step_col]

sae_lora_2_df = df[[step_col, sae_lora_2_col]].copy()
sae_lora_2_df = sae_lora_2_df[sae_lora_2_df[sae_lora_2_col] != ""]
sae_lora_2_df = sae_lora_2_df.dropna(subset=[sae_lora_2_col])
sae_lora_2_df[sae_lora_2_col] = sae_lora_2_df[sae_lora_2_col].astype(float)
sae_lora_2_df[step_col] = sae_lora_2_df[step_col]

sae_ft_df = df[[step_col, sae_ft_col]].copy()
sae_ft_df = sae_ft_df[sae_ft_df[sae_ft_col] != ""]
sae_ft_df = sae_ft_df.dropna(subset=[sae_ft_col])
sae_ft_df[sae_ft_col] = sae_ft_df[sae_ft_col].astype(float)
# Convert steps to millions of tokens (each step is 1000 tokens)
sae_ft_df[step_col] = sae_ft_df[step_col]

# Plot each line separately using clean DataFrames
plt.plot(
    sae_lora_64_df[step_col],
    sae_lora_64_df[sae_lora_64_col],
    linestyle="-",
    linewidth=2,
    color="#1f77b4",
    label="SAE LoRA Rank 64",
)

plt.plot(
    sae_lora_2_df[step_col],
    sae_lora_2_df[sae_lora_2_col],
    linestyle="-",
    linewidth=2,
    color="#ff7f0e",
    label="SAE LoRA Rank 2",
)

plt.plot(
    sae_ft_df[step_col],
    sae_ft_df[sae_ft_col],
    linestyle="-",
    linewidth=2,
    color="#2ca02c",
    label="SAE Full Fine-tune",
)

# Add horizontal line for original model loss if you still want it
# Adjust this value based on your new dataset if needed
# original_model_loss = 1.9574154930812675
y_min = (
    min(
        sae_lora_64_df[sae_lora_64_col].min(),
        sae_lora_2_df[sae_lora_2_col].min(),
        sae_ft_df[sae_ft_col].min(),
        # original_model_loss,
    )
    - 0.01
)
y_max = max(
    sae_lora_64_df[sae_lora_64_col].max(),
    sae_lora_2_df[sae_lora_2_col].max(),
    sae_ft_df[sae_ft_col].max(),
    # original_model_loss,
)

# plt.axhline(
#     y=original_model_loss,
#     color="r",
#     linestyle="--",
#     linewidth=1.5,
#     label="Original Model Loss (best achievable)",
# )

# Add labels
plt.xlabel("Training Tokens (millions)", fontweight="bold", fontsize=18)
plt.ylabel("Validation Loss", fontweight="bold", fontsize=18)
# plt.title("LoRA and SAE Training Comparison", fontsize=16)

# Set custom y-axis limit
plt.autoscale(tight=True)
# Adjust the top limit if needed based on your data
plt.ylim(bottom=y_min, top=y_max)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend(frameon=True, fontsize=18)

# Increase tick label sizes
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Ensure tight layout
plt.tight_layout()

# Save the figure with the new filename
plt.savefig(
    f"{image_dir}/sae_lora_experiment.png", format="png", dpi=300, bbox_inches="tight"
)

# plt.show()
