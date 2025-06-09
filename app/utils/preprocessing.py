import os
import re
import unicodedata
from typing import Optional
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ 형태소 분석기 초기화
kiwi = Kiwi()

# ✅ 현재 파일 기준 경로 추출
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))  # law_chatbot 폴더
model_dir = os.path.join(project_root, "EXAONE-3.5-7.8B-Instruct")  # 모델 경로

# ✅ 모델 로딩
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )

# ✅ 텍스트 전처리
def clean_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ✅ 형태소 기반 키워드 추출
def extract_keywords_morph(text: str, top_k: int = 3) -> list[str]:
    tokens = kiwi.tokenize(text)
    words = [token.form for token in tokens if token.tag.startswith("NN")]
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]

# ✅ 단순 키워드 추출 (미사용 시 삭제 가능)
def extract_keywords(text: str, top_k: int = 3) -> list[str]:
    words = re.findall(r"[\w가-힣]{2,}", text)
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]

# ✅ 불용어 제거
def remove_stopwords(text: str, stopwords: Optional[set] = None) -> str:
    if stopwords is None:
        stopwords = {"입니다", "그리고", "이것", "그것", "그러나"}
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])
