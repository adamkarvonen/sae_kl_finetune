import subprocess
import itertools


def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    subprocess.run(cmd, shell=True)


# Base command template
base_cmd = "python train_lora_or_sae.py --device 0 --model_type gemma --sae_layer 12 --num_train_examples 500000 --save_model --k 20"

# Run the single SAE full finetune with rank 1
cmd = f"{base_cmd} --rank 16 --LoRA_layers sae_from_scratch --kl_percent 95"
run_command(cmd)

cmd = f"{base_cmd} --rank 16 --LoRA_layers sae_from_scratch --kl_percent 0"
run_command(cmd)
