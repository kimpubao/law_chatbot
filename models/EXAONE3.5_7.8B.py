# ⚠️ [변경 내역]
# 기존: EXAONE-3.5-7.8B-Instruct 사용
# 변경: EXAONE-3.5-2.4B-Instruct 사용 (경량 모델로 메모리 및 속도 최적화 목적)
# 변경일: 2025-06-10

import os
import json
import pandas as pd
from rdflib import Graph
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import unicodedata
from konlpy.tag import Okt
import markdown

# 1. 경로 정의 및 폴더 구조 설정
base_path = "/mnt/e/chatbot_project_data/law_chatbot_dataset"
folders = {
    "terms_json": os.path.join(base_path, "law_knowledge_base", "법령용어"),
    "ontology_json": os.path.join(base_path, "law_knowledge_base", "법령지식"),
    "ontology_nt": os.path.join(base_path, "law_knowledge_base", "법률 데이터"),
    "ontology_owl": os.path.join(base_path, "law_knowledge_base", "온톱로지_목록"),
    "relationship_json": os.path.join(base_path, "law_Knowledge_Based_Relationship_Data"),
    "feedback_log": "feedback_log.csv"
}

# 2. JSON, RDF 로딩 함수
def load_json_files(folder_path):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".json"):
                try:
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        content = json.load(f)
                        data.extend(content if isinstance(content, list) else [content])
                except Exception as e:
                    print(f"{file} JSON 로딩 실패: {e}")
    return data

def load_rdf_files(folder_path):
    graphs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nt") or file.endswith(".owl"):
                g = Graph()
                try:
                    g.parse(os.path.join(root, file), format="nt" if file.endswith(".nt") else "xml")
                    graphs.append(g)
                except Exception as e:
                    print(f"{file} RDF 로딩 실패: {e}")
    return graphs

# 3. 정보 추출 함수
def extract_triples(graphs):
    triples = []
    for g in graphs:
        for s, p, o in g:
            triples.append({"subject": str(s), "predicate": str(p), "object": str(o)})
    return pd.DataFrame(triples)

# 4. 데이터 로딩
law_triple_df = extract_triples(load_rdf_files(folders["ontology_nt"]))
terms_dict = {
    item["용어"]: item["정의"] for item in load_json_files(folders["terms_json"])
    if isinstance(item, dict) and "용어" in item and "정의" in item
}
law_qa_df = pd.DataFrame(columns=["question", "answer"])  # 빈 QA 데이터프레임

# 5. 임베딩 및 검색기 생성
embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
corpus = law_qa_df["question"].tolist() if not law_qa_df.empty else []
corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True) if corpus else None
index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension()) if corpus else None
if corpus_embeddings is not None:
    index.add(corpus_embeddings)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def search_similar_questions(user_question, top_k=5):
    if law_qa_df.empty or index is None:
        return []
    query_embedding = embedder.encode(user_question, convert_to_numpy=True)
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    candidates = [(law_qa_df.iloc[i]["question"], law_qa_df.iloc[i]["answer"]) for i in I[0]]
    scores = reranker.predict([[user_question, q] for q, _ in candidates])
    sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in sorted_pairs]

# 6. 모델 로딩
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "LGAI-EXAONE/EXAONE-3.5-2.4B-instruct" 


# 8bit 양자화
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"  # 자동으로 CPU/GPU 배분
)

def ask_exaone(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # END 토큰 ID 가져오기
    end_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")

    output = model.generate(
        **inputs,
        max_new_tokens=2048,
        eos_token_id=end_token_id,  # 종료 조건
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1 # 중복 방지
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 프롬프트 제거된 순수 응답만 추출
    if prompt in response:
        response = response.replace(prompt, "").strip()
        # markdown을 html로 렌더링
        response = markdown.markdown(response)
    return response


# 7. 보조 응답
def lookup_legal_term_definition(user_input):
    for term in terms_dict:
        if term in user_input:
            return f"'{term}'의 정의: {terms_dict[term]}"
    return None

def search_rdf_triple(user_input):
    results = []
    for _, row in law_triple_df.iterrows():
        if row["subject"] in user_input or row["object"] in user_input:
            results.append(f"{row['subject']} -[{row['predicate']}]-> {row['object']}")
        if len(results) >= 3:
            break
    return "\n".join(results) if results else None

# 8. 통합 응답
def smart_legal_chat(user_input):
    term_def = lookup_legal_term_definition(user_input)
    if term_def:
        return term_def

    rdf_info = search_rdf_triple(user_input)
    if rdf_info:
        return f"RDF 기본 관련 정보:\n{rdf_info}"

    cleaned = clean_question(user_input)
    keywords = extract_keywords_morph(cleaned)
    keyword_info = ", ".join(keywords)

    top_qas = search_similar_questions(cleaned, top_k=5)
    if top_qas:
        retrieved_q, retrieved_a = top_qas[0]
        prompt = f"사용자 질문: {cleaned}\n\n키워드: {keyword_info}\n\n기존 질문: {retrieved_q}\n\n기존 답변: {retrieved_a}\n\n이 내용을 참고해서 자세히 설명해주세요."
    else:
        prompt = f"{cleaned}에 대해 자세히 설명해주세요."

    return ask_exaone(prompt)

# 9. Feedback 저장
def save_feedback(user_question, model_answer, user_feedback):
    log = pd.DataFrame([{
        "question": user_question,
        "model_answer": model_answer,
        "feedback": user_feedback
    }])
    if os.path.exists(folders["feedback_log"]):
        existing = pd.read_csv(folders["feedback_log"])
        pd.concat([existing, log], ignore_index=True).to_csv(folders["feedback_log"], index=False)
    else:
        log.to_csv(folders["feedback_log"], index=False)

# 10. 텍스트 전처리 및 형태소 분석 기반 키워드 추출
def clean_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_keywords_morph(text: str, top_k: int = 5) -> list[str]:
    okt = Okt()
    words = [w for w, t in okt.pos(text) if t in ("Noun", "Alpha") and len(w) > 1]
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]
