from huggingface_hub import snapshot_download

# 기존 7.8B 모델 다운로드 코드
# snapshot_download(
#     repo_id="LGAI-EXAONE/EXAONE-3.5-7.8B-instruct",
#     local_dir="EXAONE-3.5-7.8B-Instruct",
#     local_dir_use_symlinks=False
# )



# 새롭게 2.4B 모델 다운로드
# snapshot_download(
#     repo_id="LGAI-EXAONE/EXAONE-3.5-2.4B-instruct",
#     local_dir="EXAONE-3.5-2.4B-Instruct",
#     local_dir_use_symlinks=False
# )



from transformers import AutoTokenizer, AutoModel
import os

# 저장할 폴더
save_dir = "./models"

# 다운로드 대상 모델들
models_to_download = {
    "klue/roberta-base": "klue_roberta_base",
    "beomi/KcELECTRA-base-v2022": "kc_electra_base_v2022"
}

# 다운로드 실행
for model_id, folder_name in models_to_download.items():
    local_path = os.path.join(save_dir, folder_name)
    print(f"🔽 다운로드 중: {model_id} → {local_path}")
    
    AutoTokenizer.from_pretrained(model_id, cache_dir=local_path)
    AutoModel.from_pretrained(model_id, cache_dir=local_path)

print("✅ 모든 모델 다운로드 완료.")