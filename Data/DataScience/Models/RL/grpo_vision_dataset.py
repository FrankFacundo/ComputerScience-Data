import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

# ------------------------------------------------------------------
# 1. Choose where you want the dataset to live on your machine
# ------------------------------------------------------------------
DATASETS_DIR = os.path.expanduser("hf_datasets")  # any writable folder
REPO_ID = "trl-internal-testing/zen-image"
CONFIG = "conversational_prompt_only"  # subset / config

# ------------------------------------------------------------------
# 2. Download (or re‑use if it’s already there)
#    snapshot_download will skip files that are unchanged.
# ------------------------------------------------------------------
local_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DATASETS_DIR,  # root folder you picked above
    local_dir_use_symlinks=False,  # store full copies instead of links
)
print(local_path)

# ------------------------------------------------------------------
# 3. Load from disk instead of the Hub
# ------------------------------------------------------------------
dataset = load_dataset(local_path, CONFIG, split="train")

print(dataset)
