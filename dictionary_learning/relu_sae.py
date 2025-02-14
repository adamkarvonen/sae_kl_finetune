import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import dictionary_learning.base_sae as base_sae


class ReluSAE(base_sae.BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.l1_penalty = None
        self.model_name = model_name
        self.d_in = d_in
        self.d_sae = d_sae
        self.hook_layer = hook_layer

    def encode(self, x: torch.Tensor):
        acts = nn.functional.relu(self.encoder(x - self.b_dec))
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return self.decoder(feature_acts) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale


def load_dictionary_learning_relu_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> ReluSAE:
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "encoder.weight",
        "decoder.weight": "decoder.weight",
        "encoder.bias": "encoder.bias",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = ReluSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["encoder.bias"].shape[0],
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    sae.l1_penalty = config["trainer"]["l1_penalty"]

    d_sae, d_in = sae.decoder.weight.data.T.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "StandardTrainer":
        sae.cfg.architecture = "standard"
    elif config["trainer"]["trainer_class"] == "PAnnealTrainer":
        sae.cfg.architecture = "p_anneal"
    elif config["trainer"]["trainer_class"] == "StandardTrainerAprilUpdate":
        sae.cfg.architecture = "standard_april_update"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        print("not normalized")

    return sae
