from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-3.5-7.8B-instruct",
    local_dir="EXAONE-3.5-7.8B-Instruct",
    local_dir_use_symlinks=False
)