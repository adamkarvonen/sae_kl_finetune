#!/bin/bash

# Function to run a Python script and check for errors
run_plot_script() {
    echo "Running $1..."
    python3 plotting/"$1"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run $1"
        exit 1
    fi
    echo "Successfully completed $1"
    echo "----------------------------------------"
}

# Run each plotting script
run_plot_script "plot_adapter_vs_full_finetune.py"
run_plot_script "plot_lora_vs_kl.py"
run_plot_script "plot_two_stage_adapter.py"
run_plot_script "plot_kl_mse_vs_kl_mse.py"
run_plot_script "plot_kl_mse_vs_kl_val_loss.py"
run_plot_script "plot_kl_vs_e2e_4_saes.py"
run_plot_script "plot_kl_vs_e2e_fig1.py"

echo "All plots have been generated successfully!" 