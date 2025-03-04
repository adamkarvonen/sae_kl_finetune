"""
Generalized script for training any our experiments.
"""

# Packages
import argparse
from typing import List
from inspect import signature
import os
from utils import get_sae_hook
from dictionary_learning import topk_sae, relu_sae

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
from utils import (
    get_target_modules,
    get_peft_model_layers,
    initialize_auth,
    hf_dataset_to_generator,
    fast_hf_dataset_to_generator,
    save_data,
    load_data,
    save_model,
    train_model,
    _get_data_filenames,
    TrainingType,
)


# TRAINING ARGS: DO NOT CHANGE
class args:
    batch_size = 1
    ctx_len = 1024
    num_val_tokens = 1_000_000  # 1_000_000
    # num_val_tokens = 10_000  # 1_000_000
    examples_per_eval = 1000  # 1000
    log_steps = 96


def main(
    model_name: str,  # Model name
    sae_path: str,  # Path to pre-trained SAE
    sae_repo: str,
    sae_from_hf: bool,  # Whether to load SAE from HF
    dataset: str,  # HF dataset name
    experiment_name: str,  # Name of experiment
    run_name: str,  # Name of run
    sae_layer: int,  # Layer of SAE to hook into
    peft_layers: List[int],  # List of peft layers
    peft_type: str,  # "attn", "mlp", "pre-mlp", "both", "gate", "up"
    num_train_examples: int,
    training_type: TrainingType,
    peft_rank: int = 64,  # peft rank
    track_evals: bool = False,  # Whether to track evals
    device: int = 0,
    save_model_file: bool = False,
    args: args = args,
    dtype=torch.bfloat16,
    use_16_bit: bool = True,
):
    torch.manual_seed(0)
    kwargs = {
        key: value
        for key, value in locals().items()
        if key in signature(main).parameters
    }

    device_name = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    # initialize_auth()

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    print("Done loading models")

    print(f"Loaded {model_name} model and tokenizer:")
    # print(model)

    sae_module = topk_sae.load_dictionary_learning_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    print("sae module", sae_module)

    model.requires_grad_(False)

    train_gen, val_dataset = fast_hf_dataset_to_generator(
        dataset, tokenizer, args, model_name=model_name
    )
    print(
        f"Loaded {num_train_examples} training examples and {len(val_dataset)} validation examples"
    )

    CE_increase, val_losses_dict, total_training_minutes_dict = load_data(**kwargs)
    print(f"Loaded CE_increase, val_losses, and total training minutes")

    CE_increase_filename = _get_data_filenames(
        model_name=model_name,
        sae_path=sae_path,
        peft_layers=peft_layers,
        peft_rank=peft_rank,
        sae_from_hf=sae_from_hf,
        num_train_examples=num_train_examples,
        use_16_bit=use_16_bit,
        training_type=training_type,
    )[0]
    print(f"CE_increase_filename: {CE_increase_filename}")
    if os.path.exists(CE_increase_filename):
        print(f"CE_increase_filename already exists: {CE_increase_filename}")
        return
    print(
        f"CE_increase_filename does not exist: {CE_increase_filename}, starting training"
    )

    # train_gen, _ = hf_dataset_to_generator(dataset, tokenizer, args)

    if training_type == TrainingType.LORA:
        sae_module.eval()
        sae_module.requires_grad_(False)
        target_modules = get_target_modules(model_name, peft_layers, peft_type)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules,
        )

        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        peft_model = peft_model.to(device)
        peft_model_layers = get_peft_model_layers(
            peft_model, model_name
        )  # varies based on the model architecture
    else:
        peft_model = model
        peft_model_layers = utils.get_model_layers(peft_model, model_name)

    if training_type == TrainingType.SAE_LORA:
        sae_module.requires_grad_(False)
        target_modules = ["encoder", "decoder"]
        lora_config = LoraConfig(
            task_type=None,
            r=peft_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules,
        )
        sae_module = get_peft_model(sae_module, lora_config)
        sae_module.print_trainable_parameters()
    elif training_type == TrainingType.SAE_FULL_FINETUNE:
        sae_module = sae_module.to(dtype=torch.float32)

    print("BASE MODEL LOSS")
    hook_handle = peft_model_layers[sae_layer].register_forward_hook(
        utils.get_norm_calculation_hook()
    )
    base_loss, norm = utils.evaluate_get_norm(model, val_dataset)
    # base_loss = 0
    print(f"Base loss: {base_loss:.4f}")

    print(f"norm: {norm}")

    NORM_SCALE = torch.sqrt(norm).item()
    print(f"norm scale: {NORM_SCALE}")
    sae_module.scale_biases(1 / NORM_SCALE)
    sae_module.norm_scale = NORM_SCALE

    if hook_handle:
        hook_handle.remove()

    hook_handle = None
    if sae_path:
        print(f"Registering SAE hook (rank {peft_rank})")
        hook_handle = peft_model_layers[sae_layer].register_forward_hook(
            get_sae_hook(sae_module, tokenizer, sae_from_hf)
        )

    print("INITIAL PEFT MODEL LOSS")
    initial_loss, initial_l0 = utils.evaluate_get_l0(peft_model, val_dataset)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Initial l0: {initial_l0}")

    val_losses, total_training_minutes = train_model(
        peft_model=peft_model,
        sae=sae_module,
        train_gen=train_gen,
        val_dataset=val_dataset,
        args=args,
        rank=peft_rank,
        project_name=experiment_name,
        run_name=run_name,
        initial_loss=initial_loss,
        base_loss=base_loss,
        track_evals=track_evals,
        training_type=training_type,
        target_l0=initial_l0,
    )
    converged_loss = val_losses[-1]

    sae_module.scale_biases(NORM_SCALE)

    if hook_handle:
        hook_handle.remove()

    CE_increase = {
        "base": base_loss,
        "initial": initial_loss,
        "converged": converged_loss,
        "difference": initial_loss - converged_loss,
    }
    val_losses_dict = val_losses
    total_training_minutes_dict = total_training_minutes

    print(f"Rank {peft_rank}:")
    print(f"  - Initial Loss: {initial_loss:.4f}")
    print(f"  - Converged Loss: {converged_loss:.4f}")
    print(f"  - Loss Increase: {CE_increase['difference']:.4f}")
    print(f"Updated and saved CE_increase data")

    save_data(CE_increase, val_losses_dict, total_training_minutes_dict, **kwargs)

    if not save_model_file:
        print()
    elif (
        training_type == TrainingType.SAE_FULL_FINETUNE
        or training_type == TrainingType.SAE_LORA
    ):
        utils.save_sae(sae_module, peft_rank, **kwargs)
    else:
        save_model(peft_model, peft_rank, **kwargs)

    del peft_model, model, tokenizer
    if sae_path:
        del sae_module


if __name__ == "__main__":
    """python train_lora_or_sae.py --device 0 --model_type "pythia" --sae_layer 8 --rank 16 --num_train_examples 15000 --save_model --trainer_id 2 --LoRA_layers sae_lora
    python train_lora_or_sae.py --device 0 --model_type "gemma" --sae_layer 12 --rank 16 --num_train_examples 15000 --save_model --trainer_id 2 --LoRA_layers sae_lora
    """
    # Run Experiment Args
    parser = argparse.ArgumentParser(
        description="Arguments related to running experiment"
    )
    parser.add_argument("--device", type=int, required=True, help="CUDA device index")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gemma", "llama", "pythia"],
        help="Type of model",
    )
    parser.add_argument("--sae_layer", type=int, default=12, help="SAE layer")
    parser.add_argument(
        "--LoRA_layers",
        type=str,
        choices=["all", "after", "single", "sae_lora", "sae_full_finetune"],
        default="all",
        help="Which layers to apply LoRA to",
    )
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank")
    parser.add_argument(
        "--save_model", action="store_true", help="Whether to save the model"
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        help="Number of training examples",
        choices=[30, 150, 300, 3_000, 15_000, 30_000, 100_000],
        required=True,
    )
    parser.add_argument(
        "--checkpoint_percent",
        type=int,
        help="Train on a specific checkpoint percent. If None, train on all checkpoints",
    )
    parser.add_argument(
        "--trainer_id",
        type=int,
        help="Train on a specific trainer_id. If None, train on all trainer ids",
    )

    parsed_args = parser.parse_args()
    for arg_name, arg_value in vars(parsed_args).items():
        setattr(args, arg_name, arg_value)

    layer = args.sae_layer
    rank = args.rank
    # percents = [args.checkpoint_percent] if args.checkpoint_percent else range(10, 101, 10)
    trainer_ids = [args.trainer_id] if args.trainer_id is not None else range(1, 4)

    if args.LoRA_layers == "sae_lora":
        training_type = TrainingType.SAE_LORA
    elif args.LoRA_layers == "sae_full_finetune":
        training_type = TrainingType.SAE_FULL_FINETUNE
    else:
        training_type = TrainingType.LORA

    # dataset_name = "togethercomputer/RedPajama-Data-V2"
    dataset_name = "monology/pile-uncopyrighted"

    if args.model_type == "gemma":
        model_name = "google/gemma-2-2b"
        sae_repo = "canrager/saebench_gemma-2-2b_width-2pow16_date-0107"
        sae_path_template = "gemma-2-2b_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_{trainer_id}/ae.pt"
        LoRA_layers = (
            list(range(26))
            if args.LoRA_layers == "all"
            else [layer + 1]
            if args.LoRA_layers == "single"
            else list(range(layer + 1, 26))
            if args.LoRA_layers == "after"
            else []
        )
        dtype = torch.bfloat16
        use_16_bit = True
        args.batch_size = 8
    elif args.model_type == "pythia":
        model_name = "EleutherAI/pythia-160m-deduped"
        sae_repo = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108"
        sae_path_template = "Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_{trainer_id}/ae.pt"
        LoRA_layers = (
            list(range(12))
            if args.LoRA_layers == "all"
            else [layer + 1]
            if args.LoRA_layers == "single"
            else list(range(layer + 1, 12))
            if args.LoRA_layers == "after"
            else []
        )
        dtype = torch.float32
        use_16_bit = False
        args.batch_size = 8
    elif args.model_type == "llama":
        raise ValueError
        model_name = "meta-llama/Llama-3.2-1B"
        sae_path_template = "saved_saes/Llama-3.2-1B/normal/expansion_8_L0_64-{pct}pct/model.layers.{layer}"
        LoRA_layers = (
            list(range(16)) if args.LoRA_layers == "all" else list(range(layer + 1, 16))
        )

    for trainer_id in trainer_ids:
        sae_path = sae_path_template.format(trainer_id=trainer_id, layer=layer)

        main(
            model_name=model_name,
            sae_path=sae_path,
            sae_repo=sae_repo,
            sae_from_hf=False,
            dataset=dataset_name,
            experiment_name=f"{args.model_type}_LoRA",
            run_name=f"layer_{layer}_rank_{rank}_{training_type.value}_trainer_id_{trainer_id}_normed",
            sae_layer=layer,
            peft_layers=LoRA_layers,
            peft_type="both",
            peft_rank=rank,
            num_train_examples=args.num_train_examples,
            track_evals=True,
            device=args.device,
            save_model_file=args.save_model,
            dtype=dtype,
            use_16_bit=use_16_bit,
            training_type=training_type,
        )
