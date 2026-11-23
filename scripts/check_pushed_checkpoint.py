"""Check that every file of a local checkpoint step is on the Hub with the same size.

Usage: uv run python scripts/check_pushed_checkpoint.py <hf_repo> <local_step_dir> <step>
Exits 0 if the Hub copy is complete, 1 (listing the bad files) otherwise. Used by push_checkpoints.sh after each upload.
"""

import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

repo, local_dir, step = sys.argv[1:]
remote = {
    f.path: f.size for f in HfApi().list_repo_tree(repo, path_in_repo=step, recursive=True) if isinstance(f, RepoFile)
}
bad = []
for root, _, files in os.walk(local_dir):
    for name in files:
        path = os.path.join(root, name)
        key = f"{step}/{os.path.relpath(path, local_dir)}"
        if remote.get(key) != os.path.getsize(path):
            bad.append(key)
for key in bad:
    print(f"missing or wrong size on the Hub: {key}")
sys.exit(1 if bad else 0)
