import re
import unicodedata
from typing import Optional

def clean_question(text: str) -> str:
    """사용자 질문 전처리: 불필요한 공백, 특수문자 제거"""
    text = unicodedata.normalize("NFKC", text)  # 유니코드 정규화
    text = re.sub(r"[\t\n\r]+", " ", text)   # 줄바꿈, 탭 제거
    text = re.sub(r"\s+", " ", text).strip()    # 중복 공백 제거
    return text

def extract_keywords(text: str, top_k: int = 3) -> list[str]:
    """간단한 키워드 추출 (형태소 분석기 미사용 버전)"""
    # 한글/영문 단어만 추출 (기본 필터)
    words = re.findall(r"[\w가-힣]{2,}", text)
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]

def remove_stopwords(text: str, stopwords: Optional[set] = None) -> str:
    """불용어 제거 (커스텀 리스트 사용 가능)"""
    if stopwords is None:
        stopwords = {"입니다", "그리고", "이것", "그것", "그러나"}
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])