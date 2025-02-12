import torch

from dictionary_learning import topk_sae

repo_id = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108"
filename = "TopK_pythia-160m-deduped__0108/resid_post_layer_8/trainer_2/ae.pt"
layer = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

model_name = "EleutherAI/pythia-160m-deduped"
hook_name = f"blocks.{layer}.hook_resid_post"

sae = topk_sae.load_dictionary_learning_topk_sae(
    repo_id,
    filename,
    model_name,
    device,  # type: ignore
    dtype,
    layer=layer,
)
sae.test_sae(model_name)
