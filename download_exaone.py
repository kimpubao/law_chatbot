from huggingface_hub import snapshot_download

# 기존 7.8B 모델 다운로드 코드
# snapshot_download(
#     repo_id="LGAI-EXAONE/EXAONE-3.5-7.8B-instruct",
#     local_dir="EXAONE-3.5-7.8B-Instruct",
#     local_dir_use_symlinks=False
# )

# 새롭게 3.5B 모델 다운로드
snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-3.5-2.4B-instruct",
    local_dir="EXAONE-3.5-2.4B-Instruct",
    local_dir_use_symlinks=False
)