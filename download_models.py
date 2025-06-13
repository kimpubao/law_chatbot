# from huggingface_hub import snapshot_download

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


from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 저장 경로
save_dir_polyglot = "models/polyglot-ko-1.3b"
save_dir_kogpt2 = "models/kogpt2-base-v2"

# 모델 ID
polyglot_id = "EleutherAI/polyglot-ko-1.3b"
kogpt2_id = "skt/kogpt2-base-v2"

# Polyglot-Ko 1.3B 다운로드
print("🔽 polyglot-ko-1.3b 다운로드 중...")
AutoTokenizer.from_pretrained(polyglot_id).save_pretrained(save_dir_polyglot)
AutoModelForCausalLM.from_pretrained(polyglot_id).save_pretrained(save_dir_polyglot)
print("✅ polyglot-ko-1.3b 저장 완료:", save_dir_polyglot)

# KoGPT2 다운로드
print("🔽 kogpt2-base-v2 다운로드 중...")
AutoTokenizer.from_pretrained(kogpt2_id).save_pretrained(save_dir_kogpt2)
AutoModelForCausalLM.from_pretrained(kogpt2_id).save_pretrained(save_dir_kogpt2)
print("✅ kogpt2-base-v2 저장 완료:", save_dir_kogpt2)
