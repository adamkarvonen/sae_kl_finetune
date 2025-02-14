import subprocess
import itertools


def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    subprocess.run(cmd, shell=True)


# Base command template
base_cmd = "python train_lora_or_sae.py --device 0 --model_type pythia --sae_layer 8 --num_train_examples 15000 --save_model"

# Run the single SAE full finetune with rank 1
# cmd = f"{base_cmd} --rank 16 --LoRA_layers sae_full_finetune"
# run_command(cmd)

# Parameters to iterate over
# ranks = [16, 64]
# lora_layers = ["single"]

# # Run LoRA variations
# for rank, layer_type in itertools.product(ranks, lora_layers):
#     cmd = f"{base_cmd} --rank {rank} --LoRA_layers {layer_type}"
#     run_command(cmd)

trainer_ids = [5, 4, 3, 2, 1]
# ranks = [16, 64]

# 174.0

for trainer_id in trainer_ids:
    cmd = (
        f"{base_cmd} --rank 1 --LoRA_layers sae_full_finetune --trainer_id {trainer_id}"
    )
    run_command(cmd)


# lora_layers = ["all", "after"]
# for trainer_id, rank, layer_type in itertools.product(trainer_ids, ranks, lora_layers):
#     cmd = (
#         f"{base_cmd} --rank {rank} --LoRA_layers {layer_type} --trainer_id {trainer_id}"
#     )
#     run_command(cmd)

print("\nAll training runs completed!")
