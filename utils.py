import os
from dataclasses import dataclass
from itertools import islice
from typing import Generator, List, Tuple, Optional, Any, Dict
import time
import pickle
from enum import Enum
import einops
import gc

import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm.auto import tqdm
from contextlib import contextmanager

from eleuther_sae.sae.data import chunk_and_tokenize
import dictionary_learning.topk_sae as topk_sae


########################################
### MODEL ARCHITECTURE CONFIGS ###
########################################


class TrainingType(Enum):
    LORA = "lora"  # Standard LoRA training
    SAE_LORA = "sae_lora"  # SAE with LoRA training
    SAE_FULL_FINETUNE = "sae_full_finetune"  # SAE with full model fine-tuning


@dataclass
class ModelArchConfig:
    """Configuration for a model architecture"""

    attn_path_template: str
    mlp_path_template: str
    attn_modules: List[str]
    mlp_modules: List[str]


class ModelConfigs:
    """Configurations for different model architectures"""

    LLAMA = ModelArchConfig(
        attn_path_template="model.layers.{}.self_attn.{}",
        mlp_path_template="model.layers.{}.mlp.{}",
        attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        mlp_modules=["gate_proj", "up_proj", "down_proj"],
    )

    GEMMA = ModelArchConfig(
        attn_path_template="model.layers.{}.self_attn.{}",
        mlp_path_template="model.layers.{}.mlp.{}",
        attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        mlp_modules=["gate_proj", "up_proj", "down_proj"],
    )

    GPT2 = ModelArchConfig(
        attn_path_template="transformer.h.{}.attn.{}",
        mlp_path_template="transformer.h.{}.mlp.{}",
        attn_modules=["c_attn", "c_proj"],
        mlp_modules=["c_fc", "c_proj"],
    )

    PYTHIA = ModelArchConfig(
        attn_path_template="gpt_neox.layers.attention{}",
        mlp_path_template="gpt_neox.layers.{}.mlp.{}",
        attn_modules=["query_key_value", "dense"],
        mlp_modules=["dense_h_to_4h", "dense_4h_to_h"],
    )

    @classmethod
    def get_config(cls, model_name: str) -> ModelArchConfig:
        """Get the configuration for a specific model"""
        if "llama" in model_name.lower():
            return cls.LLAMA
        elif "gemma" in model_name.lower():
            return cls.GEMMA
        elif "gpt2" in model_name.lower():
            return cls.GPT2
        elif "pythia" in model_name.lower():
            return cls.PYTHIA
        else:
            raise ValueError(f"Unsupported model: {model_name}")


def get_target_modules(
    model_name: str, peft_layers: List[int], peft_type: str
) -> List[str]:
    """Get target modules for PEFT configuration"""

    config = ModelConfigs.get_config(model_name)

    def get_attn_modules():
        return [
            config.attn_path_template.format(i, module)
            for i in peft_layers
            for module in config.attn_modules
        ]

    def get_mlp_modules():
        return [
            config.mlp_path_template.format(i, module)
            for i in peft_layers
            for module in config.mlp_modules
        ]

    def get_pre_mlp_modules():
        return [
            config.mlp_path_template.format(i, module)
            for i in peft_layers
            for module in ["gate_proj", "up_proj"]
        ]

    peft_types = {
        "attn": get_attn_modules,
        "mlp": get_mlp_modules,
        "pre-mlp": get_pre_mlp_modules,
        "both": lambda: get_attn_modules() + get_mlp_modules(),
        "gate": lambda: [
            config.mlp_path_template.format(i, "gate_proj") for i in peft_layers
        ],
        "up": lambda: [
            config.mlp_path_template.format(i, "up_proj") for i in peft_layers
        ],
    }

    if peft_type not in peft_types:
        raise ValueError(f"Invalid peft_type: {peft_type}")

    return peft_types[peft_type]()


def get_peft_model_layers(peft_model, model_name: str):
    """Get the appropriate layers attribute based on model architecture.

    Args:
        peft_model: The PEFT model instance
        model_name (str): Name of the model architecture

    Returns:
        nn.ModuleList: The layers of the model

    Raises:
        ValueError: If the model architecture is not supported
    """
    if any(name in model_name.lower() for name in ["gemma", "llama", "mistral"]):
        return peft_model.model.model.layers
    elif "pythia" in model_name.lower():
        return peft_model.model.gpt_neox.layers
    elif "gpt2" in model_name.lower():
        return peft_model.model.transformer.h
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def get_model_layers(model, model_name: str):
    if any(name in model_name.lower() for name in ["gemma", "llama", "mistral"]):
        return model.model.layers
    elif "pythia" in model_name.lower():
        return model.gpt_neox.layers
    elif "gpt2" in model_name.lower():
        return model.transformer.h
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def initialize_auth() -> None:
    """Initialize authentication for Hugging Face and Weights & Biases."""
    load_dotenv()

    # Hugging Face authentication
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    login(token=hf_token)

    # Weights & Biases authentication
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    wandb.login(key=wandb_key)


########################################
### DATA PROCESSING FUNCTIONS ###
########################################


def _tokenized_batch(
    generator: Generator[str, None, None], tokenizer: PreTrainedTokenizer, args: Any
) -> torch.Tensor:
    """Create a batch of tokenized texts."""
    batch = []
    try:
        while len(batch) < args.batch_size:
            next_text = next(generator)
            tokenized = tokenizer(
                next_text,
                return_tensors="pt",
                max_length=args.ctx_len,
                padding=False,
                truncation=True,
            )
            if tokenized["input_ids"].shape[1] == args.ctx_len:
                batch.append(tokenized)
    except StopIteration:
        if not batch:
            raise RuntimeError("Generator exhausted before creating a batch")

    return torch.cat([x["input_ids"] for x in batch], dim=0)


def fast_hf_dataset_to_generator(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    args: Any,
    model_name: str,
    seed: int = 42,
) -> Tuple[Generator, List[torch.Tensor]]:
    print(f"Loading dataset: {dataset_name}")

    max_seq_len = args.ctx_len
    total_train_tokens = args.num_train_examples * max_seq_len
    val_tokens = args.num_val_tokens
    total_tokens = total_train_tokens + val_tokens

    is_redpajama = dataset_name == "togethercomputer/RedPajama-Data-V2"
    is_pile = dataset_name == "monology/pile-uncopyrighted"
    text_key = "raw_content" if is_redpajama else "text"
    tokens_dir = "tokens"

    os.makedirs(tokens_dir, exist_ok=True)

    cache_file = f"{dataset_name.replace('/', '_')}_cache_{total_tokens}_tokens.pkl"
    cache_file = os.path.join(tokens_dir, cache_file)
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            subset = pickle.load(f)
    else:
        dataset = load_dataset(
            dataset_name,
            name="sample-10B" if is_redpajama else None,
            trust_remote_code=True,
            streaming=True,
            split="train",
        )

        # dataset = load_dataset(
        #     args.dataset,
        #     name="sample-10B",
        #     split="train",
        #     streaming=True,  # Enables streaming mode
        #     trust_remote_code=True,
        # )
        # Collecting approximately 200M tokens
        print("Collecting samples from streaming dataset...")
        token_count = 0
        subset = []
        pbar = tqdm(dataset, desc="Collecting samples", total=None)
        for sample in pbar:
            text = sample[text_key]
            token_count += len(text.split())  # Approximate token count using whitespace
            subset.append(sample)

            pbar.set_postfix({"tokens": f"{token_count:,}/{total_tokens:,}"})

            if token_count >= total_tokens:  # Stop when we reach 200M tokens
                break

        # Cache the collected samples
        print(f"Saving dataset cache to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(subset, f)

    # Convert to a standard Hugging Face dataset
    from datasets import Dataset

    dataset = Dataset.from_list(subset)

    dataset = dataset.shuffle(seed=seed)

    filename_id = f"{dataset_name}_{model_name}_{total_tokens}_tokens"
    cached_tokens_file = f"{filename_id}_tokens.pkl".replace("/", "_")
    cached_tokens_file = os.path.join(tokens_dir, cached_tokens_file)

    if not os.path.exists(cached_tokens_file):
        tokenized = chunk_and_tokenize(
            dataset, tokenizer, text_key=text_key, max_seq_len=max_seq_len
        )
        print(f"Saving {cached_tokens_file} to disk")
        with open(cached_tokens_file, "wb") as f:
            pickle.dump(tokenized, f)
    else:
        print(f"Loading tokens from {cached_tokens_file}")

        with open(cached_tokens_file, "rb") as f:
            tokenized = pickle.load(f)

    print(tokenized["input_ids"].shape)

    total_cutoff = (total_tokens + max_seq_len - 1) // max_seq_len
    train_cutoff = (total_train_tokens + max_seq_len - 1) // max_seq_len
    val_cutoff = total_cutoff - train_cutoff

    # Get validation data as tensors (typically smaller)
    subset_val_data = tokenized.select(range(val_cutoff))

    # Create generator for training data that yields batches
    def train_gen():
        batch_examples = []
        for i in range(val_cutoff, val_cutoff + train_cutoff):
            batch_examples.append(tokenized[i]["input_ids"])
            if len(batch_examples) == args.batch_size:
                yield torch.stack(batch_examples)
                batch_examples = []
        # Yield any remaining examples in the final batch
        if batch_examples:
            yield torch.stack(batch_examples)

    # Convert validation data to batched tensors
    val_tensors = []
    for i in range(0, len(subset_val_data), args.batch_size):
        batch_indices = range(i, min(i + args.batch_size, len(subset_val_data)))
        batch = torch.stack([subset_val_data[j]["input_ids"] for j in batch_indices])
        val_tensors.append(batch)

    print(
        f"Created {len(val_tensors)} validation batches of size up to {args.batch_size}"
    )
    return train_gen(), val_tensors


def hf_dataset_to_generator(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    args: Any,
    split: str = "train",
    streaming: bool = True,
    seed: int = 42,
) -> Tuple[Generator, List[torch.Tensor]]:
    """
    Process a Hugging Face dataset into training generator and validation tensors.
    Returns:
        Tuple[Generator, List[torch.Tensor]]: A tuple containing:
            - Generator that yields batched training examples
            - List of validation tensors (batched)
    """
    print(f"Loading dataset: {dataset_name}")
    is_redpajama = dataset_name == "togethercomputer/RedPajama-Data-V2"
    is_pile = dataset_name == "monology/pile-uncopyrighted"
    dataset = load_dataset(
        dataset_name,
        name="sample-10B" if is_redpajama else None,
        data_files=["train/00.jsonl.zst"] if is_pile else None,
        trust_remote_code=True,
    )

    dataset = dataset.shuffle(seed=42)
    max_seq_len = args.ctx_len
    tokenized = chunk_and_tokenize(
        dataset,
        tokenizer,
        text_key="raw_content" if is_redpajama else "text",
        max_seq_len=max_seq_len,
    )["train"]

    total_train_tokens = args.num_train_examples * max_seq_len
    val_tokens = args.num_val_tokens
    total_tokens = total_train_tokens + val_tokens

    total_cutoff = (total_tokens + max_seq_len - 1) // max_seq_len
    train_cutoff = (total_train_tokens + max_seq_len - 1) // max_seq_len
    val_cutoff = total_cutoff - train_cutoff

    # Get validation data as tensors (typically smaller)
    subset_val_data = tokenized.select(range(val_cutoff))

    # Create generator for training data that yields batches
    def train_gen():
        batch_examples = []
        for i in range(val_cutoff, val_cutoff + train_cutoff):
            batch_examples.append(tokenized[i]["input_ids"])
            if len(batch_examples) == args.batch_size:
                yield torch.stack(batch_examples)
                batch_examples = []
        # Yield any remaining examples in the final batch
        if batch_examples:
            yield torch.stack(batch_examples)

    # Convert validation data to batched tensors
    val_tensors = []
    for i in range(0, len(subset_val_data), args.batch_size):
        batch_indices = range(i, min(i + args.batch_size, len(subset_val_data)))
        batch = torch.stack([subset_val_data[j]["input_ids"] for j in batch_indices])
        val_tensors.append(batch)

    print(
        f"Created {len(val_tensors)} validation batches of size up to {args.batch_size}"
    )
    return train_gen(), val_tensors


########################################
### SAVE DATA FUNCTIONS ###
########################################


def _get_data_filenames(
    model_name: str,
    sae_path: Optional[str],
    peft_layers: List[int],
    sae_from_hf: bool,
    peft_rank: int,
    num_train_examples: int,
    use_16_bit: bool,
    training_type: TrainingType,
) -> Tuple[str, str, str]:
    """
    Helper function to generate filenames for CE increase and validation losses data.
    """
    num_train_examples_in_thousands = num_train_examples // 1000
    model_name = model_name.split("/")[-1]

    if sae_path:
        if not sae_from_hf:
            sae_path = sae_path.split("/")[-2]
        else:
            parts = sae_path.split("/", 1)
            sae_path = f"{parts[1].replace('/', '_')}"

    if training_type == TrainingType.SAE_LORA:
        peft_range = training_type.value
    elif training_type == TrainingType.SAE_FULL_FINETUNE:
        peft_range = training_type.value
    else:
        peft_range = (
            f"{peft_layers[0]}-{peft_layers[-1]}"
            if len(peft_layers) > 1
            else peft_layers[0]
        )

    # Build base paths
    base_path = f"data/scaling" if sae_from_hf else f"data/TopK"

    ce_base_path = f"{base_path}/CE_increase/{model_name}/{sae_path}"
    val_base_path = f"{base_path}/val_loss/{model_name}/{sae_path}"
    time_base_path = f"{base_path}/time/{model_name}/{sae_path}"

    CE_increase_filename = f"{ce_base_path}/peft_{peft_range}_rank_{peft_rank}_CE_increase_{num_train_examples_in_thousands}k.json"
    val_losses_filename = f"{val_base_path}/peft_{peft_range}_rank_{peft_rank}_val_losses_{num_train_examples_in_thousands}k.json"
    time_filename = f"{time_base_path}/peft_{peft_range}_rank_{peft_rank}_time_{num_train_examples_in_thousands}k.json"

    return CE_increase_filename, val_losses_filename, time_filename


def save_data(
    CE_increase: dict,
    val_losses_dict: dict,
    total_training_minutes_dict: dict,
    **kwargs,
) -> None:
    """
    Save CE increase and validation losses data to json files.
    """
    CE_increase_filename, val_losses_filename, time_filename = _get_data_filenames(
        model_name=kwargs["model_name"],
        sae_path=kwargs["sae_path"],
        peft_layers=kwargs["peft_layers"],
        sae_from_hf=kwargs["sae_from_hf"],
        peft_rank=kwargs["peft_rank"],
        num_train_examples=kwargs["num_train_examples"],
        use_16_bit=kwargs["use_16_bit"],
        training_type=kwargs["training_type"],
    )

    os.makedirs(os.path.dirname(CE_increase_filename), exist_ok=True)
    os.makedirs(os.path.dirname(val_losses_filename), exist_ok=True)
    os.makedirs(os.path.dirname(time_filename), exist_ok=True)

    with open(CE_increase_filename, "w") as f:
        json.dump(CE_increase, f, indent=4)

    with open(val_losses_filename, "w") as f:
        json.dump(val_losses_dict, f, indent=4)

    with open(time_filename, "w") as f:
        json.dump(total_training_minutes_dict, f, indent=4)

    print(f"Saved CE increase data to {CE_increase_filename}")
    print(f"Saved validation losses to {val_losses_filename}")
    print(f"Saved total training minutes to {time_filename}")


def load_data(**kwargs) -> Tuple[dict, dict, dict]:
    """
    Load CE increase and validation losses data from json files.
    """
    CE_increase_filename, val_losses_filename, time_filename = _get_data_filenames(
        model_name=kwargs["model_name"],
        sae_path=kwargs["sae_path"],
        peft_layers=kwargs["peft_layers"],
        sae_from_hf=kwargs["sae_from_hf"],
        peft_rank=kwargs["peft_rank"],
        num_train_examples=kwargs["num_train_examples"],
        use_16_bit=kwargs["use_16_bit"],
        training_type=kwargs["training_type"],
    )

    try:
        with open(CE_increase_filename, "r") as f:
            CE_increase = json.load(f)

        with open(val_losses_filename, "r") as f:
            val_losses_dict = json.load(f)

        with open(time_filename, "r") as f:
            total_training_minutes_dict = json.load(f)

        print(f"Loaded CE increase data from {CE_increase_filename}")
        print(f"Loaded validation losses from {val_losses_filename}")

        return CE_increase, val_losses_dict, total_training_minutes_dict

    except FileNotFoundError:
        print(f"Could not find data files. Creating new files...")
        return {}, {}, {}
    except json.JSONDecodeError:
        print(f"Error parsing JSON files. Creating new files...")
        return {}, {}, {}


def save_model(peft_model, rank, **kwargs) -> None:
    sae_path = kwargs["sae_path"]
    model_name = kwargs["model_name"]
    peft_layers = kwargs["peft_layers"]
    peft_type = kwargs["peft_type"]
    sae_from_hf = kwargs["sae_from_hf"]
    training_type = kwargs["training_type"]

    if sae_path:
        if not sae_from_hf:
            sae_path = sae_path.split("/")[-2]
        else:
            parts = sae_path.split("/", 1)
            sae_path = f"{parts[1].replace('/', '_')}"

    model_name = model_name.split("/")[-1]

    if training_type == TrainingType.SAE_FULL_FINETUNE:
        raise ValueError("This is only for loras")
    elif training_type == TrainingType.SAE_LORA:
        peft_range = training_type.value
    else:
        peft_range = (
            f"{peft_layers[0]}-{peft_layers[-1]}"
            if len(peft_layers) > 1
            else peft_layers[0]
        )

    base_path = f"saved_models/{model_name}"
    base_path = f"{base_path}/{sae_path}" if sae_path else f"{base_path}/base"
    save_dir = f"{base_path}/peft_{peft_range}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"rank_{rank}")
    os.makedirs(model_path, exist_ok=True)
    peft_model.save_pretrained(model_path)
    print(f"Saved model to {model_path}")


def save_sae(sae, rank, **kwargs) -> None:
    sae_path = kwargs["sae_path"]
    model_name = kwargs["model_name"]
    peft_layers = kwargs["peft_layers"]
    peft_type = kwargs["peft_type"]
    sae_from_hf = kwargs["sae_from_hf"]
    training_type = kwargs["training_type"]

    if sae_path:
        if not sae_from_hf:
            sae_path = sae_path.split("/")[-2]
        else:
            parts = sae_path.split("/", 1)
            sae_path = f"{parts[1].replace('/', '_')}"

    model_name = model_name.split("/")[-1]

    if (
        training_type == TrainingType.SAE_LORA
        or training_type == TrainingType.SAE_FULL_FINETUNE
    ):
        peft_range = training_type.value
    else:
        raise ValueError("This is only for saving SAEs")

    base_path = f"saved_models/{model_name}"
    base_path = f"{base_path}/{sae_path}" if sae_path else f"{base_path}/base"
    save_dir = f"{base_path}/peft_{peft_range}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"rank_{rank}")
    os.makedirs(model_path, exist_ok=True)

    final = {k: v.cpu() for k, v in sae.state_dict().items()}
    torch.save(final, os.path.join(model_path, "ae.pt"))
    print(f"Saved model to {model_path}")


###############################################
### TRAIN AND EVALUATION FUNCTIONS ###
###############################################
@contextmanager
def wandb_session(project_name: str, run_name: str, config: Dict[str, Any]):
    """Context manager for wandb session"""
    try:
        wandb.init(project=project_name, name=run_name, config=config)
        yield
    finally:
        wandb.finish()


def evaluate(
    model: PreTrainedModel,
    val_dataset: List[torch.Tensor],
) -> float:
    """Evaluate model on validation dataset"""
    device = model.device
    model.eval()

    val_loss = 0
    total_examples = 0

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        val_loop = tqdm(val_dataset, leave=True, desc="Validation")
        try:
            for val_batch in val_loop:
                val_inputs = val_batch.to(device)
                val_targets = val_inputs.clone()
                batch_size = val_inputs.size(0)

                GlobalSAE.current_batch = val_inputs.detach()
                val_outputs = model(val_inputs, labels=val_targets)
                val_loss += val_outputs.loss.item() * batch_size
                total_examples += batch_size

                # Cleanup
                # del val_inputs, val_targets, val_outputs
                # torch.cuda.empty_cache()

                val_loop.set_description_str(
                    f"Validation Loss: {val_loss / total_examples:.4f}"
                )
        finally:
            val_loop.close()

    return val_loss / total_examples


def train_model(
    peft_model: PreTrainedModel,
    sae: topk_sae.TopKSAE,
    train_gen: Generator[str, None, None],
    val_dataset: List[torch.Tensor],
    args: Any,
    rank: int,
    project_name: str,
    run_name: str,
    training_type: TrainingType,
    gradient_accumulation_steps: int,
    initial_loss: Optional[float] = None,
    base_loss: Optional[float] = None,
    track_evals: bool = True,
    mse_only: bool = False,
    start_lr: float = 5e-5,
) -> Tuple[List[float], List[float]]:
    """Train the model using KL divergence loss with linear LR decay"""
    print("Training model with KL divergence loss")
    device = peft_model.device

    sae_only = False
    if (
        training_type == TrainingType.SAE_FULL_FINETUNE
        or training_type == TrainingType.SAE_LORA
    ):
        sae_only = True

    args_config = {
        attr: getattr(args, attr)
        for attr in dir(args)
        if not callable(getattr(args, attr)) and not attr.startswith("__")
    }

    # Initialize optimizer with start_lr
    if training_type == TrainingType.SAE_FULL_FINETUNE:
        optimizer = optim.AdamW(sae.parameters(), lr=start_lr)
    elif training_type == TrainingType.SAE_LORA:
        optimizer = optim.AdamW(sae.parameters(), lr=start_lr)
    else:
        optimizer = optim.AdamW(peft_model.parameters(), lr=start_lr)
        peft_model.train()

    def lr_lambda(current_step: int) -> float:
        progress = current_step / args.num_train_examples
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses = []
    val_losses = [initial_loss] if initial_loss is not None else []
    total_training_minutes_list = [0] if initial_loss is not None else []
    total_examples = 0

    total_training_minutes = 0

    with wandb_session(project_name, run_name, args_config):
        if initial_loss is not None:
            wandb.log(
                {
                    "examples_processed": 0,
                    "val_loss": initial_loss,
                    "base_loss": base_loss,
                    "total_training_minutes": 0,
                    "training_minutes_between_evals": 0,
                },
                step=total_examples,
            )

        total_loss = 0
        examples_since_last_eval = 0

        train_loop = tqdm(desc="Training", total=args.num_train_examples)
        training_time_between_evals = 0
        optimizer.zero_grad()  # Move initial gradient zeroing outside the loop
        accumulated_steps = 0

        try:
            while total_examples < args.num_train_examples:
                train_step_start = time.time()

                inputs = next(train_gen).to(device)
                batch_size = inputs.size(0)

                GlobalSAE.current_batch = inputs.detach()
                if mse_only:
                    GlobalSAE.use_sae = True
                    peft_model(inputs.to(peft_model.device))
                    kl_loss = 0
                else:
                    with torch.no_grad():
                        GlobalSAE.use_sae = False

                        if sae_only:
                            base_outputs = peft_model(inputs.to(peft_model.device))
                        else:
                            with peft_model.disable_adapter():
                                base_outputs = peft_model(inputs.to(peft_model.device))

                        base_logits = base_outputs.logits
                        base_probs = torch.nn.functional.softmax(
                            base_logits, dim=-1
                        ).to(peft_model.device)

                    GlobalSAE.use_sae = True
                    peft_outputs = peft_model(inputs)
                    peft_logits = peft_outputs.logits
                    peft_log_probs = torch.nn.functional.log_softmax(
                        peft_logits, dim=-1
                    )

                    # Calculate KL divergence loss
                    kl_loss = torch.nn.functional.kl_div(
                        peft_log_probs,
                        base_probs,
                        reduction="batchmean",
                        log_target=False,
                    )

                if sae_only:
                    mse_loss = GlobalSAE.reconstruction_loss

                    if mse_only:
                        alpha_kl = 0
                        loss = mse_loss
                    else:
                        # Reconstruction loss matches original mse loss scale so an optional sparsity penalty stays relevant
                        alpha_kl = (mse_loss / (kl_loss + 1e-8)).detach()
                        loss = (kl_loss * alpha_kl + mse_loss) * 0.5

                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    accumulated_steps += 1
                    if accumulated_steps == gradient_accumulation_steps:
                        if training_type == TrainingType.SAE_FULL_FINETUNE:
                            sae.decoder.weight.grad = (
                                remove_gradient_parallel_to_decoder_directions(
                                    sae.decoder.weight,
                                    sae.decoder.weight.grad,
                                    sae.d_in,
                                    sae.d_sae,
                                )
                            )
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step(total_examples)
                        optimizer.zero_grad()
                        accumulated_steps = 0

                        # clip grad norm and remove grads parallel to decoder directions
                        if training_type == TrainingType.SAE_FULL_FINETUNE:
                            # Make sure the decoder is still unit-norm
                            sae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                                sae.decoder.weight, sae.d_in, sae.d_sae
                            )

                    if examples_since_last_eval % args.log_steps == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        wandb.log(
                            {
                                "mse_loss": mse_loss,
                                "kl_loss": kl_loss,
                                "alpha_kl": alpha_kl,
                                "learning_rate": current_lr,
                            },
                            step=total_examples,
                        )
                else:
                    loss = kl_loss
                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    accumulated_steps += 1
                    if accumulated_steps == gradient_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(
                            peft_model.parameters(), max_norm=1.0
                        )
                        optimizer.step()
                        scheduler.step(total_examples)
                        optimizer.zero_grad()
                        accumulated_steps = 0

                # Use the unscaled loss for logging
                total_loss += (loss.item() * gradient_accumulation_steps) * batch_size
                total_examples += batch_size
                examples_since_last_eval += batch_size

                training_time_between_evals += time.time() - train_step_start
                train_loop.update(batch_size)

                if track_evals and examples_since_last_eval >= args.examples_per_eval:
                    avg_train_loss = total_loss / examples_since_last_eval
                    val_loss = evaluate(peft_model, val_dataset)

                    total_training_minutes += training_time_between_evals / 60

                    train_losses.append(avg_train_loss)
                    val_losses.append(val_loss)
                    total_training_minutes_list.append(total_training_minutes)

                    wandb.log(
                        {
                            "examples_processed": total_examples,
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                            "training_minutes_between_evals": training_time_between_evals
                            / 60,
                            "total_training_minutes": total_training_minutes,
                        },
                        step=total_examples,
                    )

                    print(
                        f"\nExamples: {total_examples}, Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Training Minutes Between Evals: {training_time_between_evals / 60:.2f}, "
                        f"Total Training Minutes: {total_training_minutes:.2f}"
                    )

                    # Reset counters
                    total_loss = 0
                    examples_since_last_eval = 0
                    training_time_between_evals = 0

        finally:
            train_loop.close()

        # Final evaluation
        val_loss = evaluate(peft_model, val_dataset)
        total_training_minutes += training_time_between_evals / 60

        train_losses.append(total_loss / total_examples)
        val_losses.append(val_loss)
        total_training_minutes_list.append(total_training_minutes)

        wandb.log(
            {
                "examples_processed": total_examples,
                "train_loss": total_loss / total_examples,
                "val_loss": val_loss,
                "training_minutes_between_evals": training_time_between_evals / 60,
                "total_training_minutes": total_training_minutes,
            },
            step=total_examples,
        )

        print(
            f"Final Evaluation: Train Loss: {total_loss / total_examples:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Total Training Minutes: {total_training_minutes:.2f}"
        )

    return val_losses, total_training_minutes_list


class GlobalSAE:
    current_batch = None
    use_sae = True
    reconstruction_loss = None
    aux_loss = None
    # sparsity_loss = None # Not needed for TopK


def get_sae_hook(sae_module, tokenizer, sae_from_hf=False):
    def sae_reconstruction_hook(module, input, output):
        if not GlobalSAE.use_sae:
            return output

        original_shape = output[0].shape
        output_tensor = output[0]

        flat_output = output_tensor.reshape(-1, original_shape[-1])
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            reconstructed_output = sae_module(flat_output)
        reconstructed_output = reconstructed_output.reshape(original_shape).to(
            dtype=output_tensor.dtype
        )

        mse = torch.nn.functional.mse_loss(reconstructed_output, output_tensor.detach())
        GlobalSAE.reconstruction_loss = mse

        return (reconstructed_output,) + output[1:]

    return sae_reconstruction_hook


# The next two functions could be replaced with the ConstrainedAdam Optimizer
@torch.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: torch.nn.Parameter, activation_dim: int, d_sae: int
) -> torch.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape

    assert D == activation_dim
    assert F == d_sae

    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: torch.Tensor,
    W_dec_DF_grad: torch.Tensor,
    activation_dim: int,
    d_sae: int,
) -> torch.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape
    assert D == activation_dim
    assert F == d_sae

    normed_W_dec_DF = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)

    parallel_component = einops.einsum(
        W_dec_DF_grad,
        normed_W_dec_DF,
        "d_in d_sae, d_in d_sae -> d_sae",
    )
    W_dec_DF_grad -= einops.einsum(
        parallel_component,
        normed_W_dec_DF,
        "d_sae, d_in d_sae -> d_in d_sae",
    )
    return W_dec_DF_grad
