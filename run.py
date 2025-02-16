import subprocess
import itertools


def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    subprocess.run(cmd, shell=True)


# Base command template
base_cmd = "python train_lora_or_sae.py --device 0 --model_type gemma --sae_layer 12 --num_train_examples 15000 --save_model --trainer_id 2"

# Run the single SAE full finetune with rank 1
cmd = f"{base_cmd} --rank 16 --LoRA_layers sae_full_finetune"
run_command(cmd)

# Parameters to iterate over
ranks = [16, 64, 768]
lora_layers = [
    "adapter_only",
    "adapter_and_sae",
    "mlp_adapter_only",
    "mlp_adapter_and_sae",
]

# # Run LoRA variations
for rank, layer_type in itertools.product(ranks, lora_layers):
    cmd = f"{base_cmd} --rank {rank} --LoRA_layers {layer_type}"
    run_command(cmd)

print("\nAll training runs completed!")
