import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = "paper_data/lora_vs_kl_finetune.csv"
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
lora_all_col = all_cols[1]  # "layer_12_rank_16_lora_trainer_2_all - val_loss"
lora_after_col = all_cols[4]  # "layer_12_rank_16_lora_trainer_2_after - val_loss"
sae_ft_col = all_cols[7]  # "layer_12_rank_16_sae_full_finetune_trainer_2 - val_loss"

# Filter and create separate DataFrames for each line to avoid NaN issues
lora_all_df = df[[step_col, lora_all_col]].copy()
lora_all_df = lora_all_df[lora_all_df[lora_all_col] != ""]
lora_all_df = lora_all_df.dropna(subset=[lora_all_col])
lora_all_df[lora_all_col] = lora_all_df[lora_all_col].astype(float)

lora_after_df = df[[step_col, lora_after_col]].copy()
lora_after_df = lora_after_df[lora_after_df[lora_after_col] != ""]
lora_after_df = lora_after_df.dropna(subset=[lora_after_col])
lora_after_df[lora_after_col] = lora_after_df[lora_after_col].astype(float)

sae_ft_df = df[[step_col, sae_ft_col]].copy()
sae_ft_df = sae_ft_df[sae_ft_df[sae_ft_col] != ""]
sae_ft_df = sae_ft_df.dropna(subset=[sae_ft_col])
sae_ft_df[sae_ft_col] = sae_ft_df[sae_ft_col].astype(float)

# Plot each line separately using clean DataFrames
plt.plot(
    lora_all_df[step_col],
    lora_all_df[lora_all_col],
    linestyle="-",
    linewidth=2,
    color="#1f77b4",
    label="LoRA After Layers",
)

plt.plot(
    lora_after_df[step_col],
    lora_after_df[lora_after_col],
    linestyle="-",
    linewidth=2,
    color="#ff7f0e",
    label="LoRA All Layers",
)

plt.plot(
    sae_ft_df[step_col],
    sae_ft_df[sae_ft_col],
    linestyle="-",
    linewidth=2,
    color="#2ca02c",  # Green color for the third line
    label="SAE Fine Tune",
)

# Add horizontal line for original model loss if you still want it
# Adjust this value based on your new dataset if needed
# original_model_loss = 1.9574154930812675
y_min = (
    min(
        lora_all_df[lora_all_col].min(),
        lora_after_df[lora_after_col].min(),
        sae_ft_df[sae_ft_col].min(),
        # original_model_loss,
    )
    - 0.01
)
y_max = max(
    lora_all_df[lora_all_col].max(),
    lora_after_df[lora_after_col].max(),
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
plt.xlabel("Training Tokens (thousands)", fontweight="bold", fontsize=14)
plt.ylabel("Validation Loss", fontweight="bold", fontsize=14)
# plt.title("LoRA and SAE Training Comparison", fontsize=16)

# Set custom y-axis limit
plt.autoscale(tight=True)
# Adjust the top limit if needed based on your data
plt.ylim(bottom=y_min, top=y_max)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add legend
plt.legend(frameon=True, fontsize=12)

# Ensure tight layout
plt.tight_layout()

# Save the figure with the new filename
plt.savefig(
    f"{image_dir}/lora_sae_comparison.png", format="png", dpi=300, bbox_inches="tight"
)

# plt.show()
