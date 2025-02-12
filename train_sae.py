import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import pickle

from eleuther_sae.sae import SaeConfig, SaeTrainer, TrainConfig
from eleuther_sae.sae.data import chunk_and_tokenize
import argparse

try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False


def main(args):
    device = args.device
    train_l0 = args.train_l0
    expansion_factor = args.expansion_factor
    e2e = args.e2e
    layer = args.layer
    batch_size = args.batch_size

    MODEL = args.model
    MODEL_NAME = MODEL.split("/")[-1]

    torch.cuda.set_device(device)

    # dataset = load_dataset(
    #     args.dataset,
    #     name="sample-10B",
    #     split="train",
    #     trust_remote_code=True,
    # )

    # Check if cached dataset exists
    cache_file = "dataset_cache_200M.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            subset = pickle.load(f)
    else:
        dataset = load_dataset(
            args.dataset,
            name="sample-10B",
            split="train",
            streaming=True,  # Enables streaming mode
            trust_remote_code=True,
        )
        # Collecting approximately 200M tokens
        print("Collecting samples from streaming dataset...")
        token_count = 0
        subset = []
        target_tokens = 200_000_000
        pbar = tqdm(dataset, desc="Collecting samples", total=None)
        for sample in pbar:
            text = sample["raw_content"]
            token_count += len(text.split())  # Approximate token count using whitespace
            subset.append(sample)

            pbar.set_postfix({"tokens": f"{token_count:,}/{target_tokens:,}"})

            if token_count >= target_tokens:  # Stop when we reach 200M tokens
                break

        # Cache the collected samples
        print(f"Saving dataset cache to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(subset, f)

    # Convert to a standard Hugging Face dataset
    from datasets import Dataset

    dataset = Dataset.from_list(subset)

    dataset = dataset.shuffle(seed=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    max_seq_len = 1024

    cached_tokens_file = f"{args.dataset.replace('/', '_')}_tokens.pkl"

    if not os.path.exists(cached_tokens_file):
        tokenized = chunk_and_tokenize(
            dataset, tokenizer, text_key="raw_content", max_seq_len=max_seq_len
        )
        print(f"Saving {cached_tokens_file} to disk")
        with open(cached_tokens_file, "wb") as f:
            pickle.dump(tokenized, f)
    else:
        print(f"Loading tokens from {cached_tokens_file}")

        with open(cached_tokens_file, "rb") as f:
            tokenized = pickle.load(f)

    print(tokenized["input_ids"].shape)

    total_train_tokens = args.num_train_tokens
    val_tokens = args.num_val_tokens
    total_tokens = total_train_tokens + val_tokens

    total_cutoff = (total_tokens + max_seq_len - 1) // max_seq_len
    train_cutoff = (total_train_tokens + max_seq_len - 1) // max_seq_len
    val_cutoff = total_cutoff - train_cutoff

    subset_val_data = tokenized.select(range(val_cutoff))
    subset_train_data = tokenized.select(range(val_cutoff, val_cutoff + train_cutoff))

    print(
        "Training data --",
        "Num Training Examples:",
        len(subset_train_data),
        "Shape:",
        subset_train_data[0]["input_ids"].shape,
    )
    print(
        "Validation data --",
        "Num Validation Examples:",
        len(subset_val_data),
        "Shape:",
        subset_val_data[0]["input_ids"].shape,
    )

    gpt = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
    )

    for param in gpt.parameters():
        param.requires_grad = False

    if e2e:
        print("Training E2E SAE")
        run_name = f"{MODEL_NAME}/e2e/expansion_{expansion_factor}_L0_{train_l0}"

        cfg = TrainConfig(
            SaeConfig(k=train_l0, expansion_factor=expansion_factor),
            batch_size=batch_size,
            layers=[layer],
            wandb_project=args.project_name,
            run_name=run_name,
            e2e=e2e,
        )

        trainer = SaeTrainer(cfg, subset_train_data, subset_val_data, gpt)
        trainer.fit()

        # Clean up memory
        del trainer
        del gpt
        torch.cuda.empty_cache()
    else:
        print("Training Normal SAE")
        run_name = f"{MODEL_NAME}/normal/expansion_{expansion_factor}_L0_{train_l0}"

        cfg = TrainConfig(
            SaeConfig(k=train_l0, expansion_factor=expansion_factor),
            batch_size=batch_size,
            layers=[layer],
            wandb_project=args.project_name,
            run_name=run_name,
        )

        trainer = SaeTrainer(cfg, subset_train_data, subset_val_data, gpt)
        trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True, help="GPU device ID")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gemma", "llama"],
        help="Which LLM to train SAE on",
    )
    parser.add_argument("--layer", type=int, default=12, help="Layer to train")
    parser.add_argument("--e2e", action="store_true", help="Whether to train e2e")
    parser.add_argument(
        "--num_train_tokens",
        type=int,
        required=True,
        choices=[200_000_000, 2_000_000_000, 4_000_000_000],
        help="Number of training tokens",
    )
    parser.add_argument(
        "--num_val_tokens", type=int, default=1_000_000, help="Number of validation tokens"
    )

    args = parser.parse_args()

    MODEL = "google/gemma-2-2b" if args.model_type == "gemma" else "meta-llama/Llama-3.2-1B"
    dataset = "togethercomputer/RedPajama-Data-V2"
    project_name = MODEL.split("/")[-1] + "_e2e_SAE" if args.e2e else MODEL.split("/")[-1] + "_SAE"
    batch_size = 1

    class Arguments:
        def __init__(self):
            self.device = args.device
            self.train_l0 = 64
            self.expansion_factor = 8
            self.e2e = args.e2e
            self.model = MODEL
            self.dataset = dataset
            self.batch_size = batch_size
            self.project_name = project_name
            self.layer = args.layer
            self.num_train_tokens = args.num_train_tokens
            self.num_val_tokens = args.num_val_tokens

    main(Arguments())
