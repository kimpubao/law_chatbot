# ⚠️ [변경 내역]
# 기존: EXAONE-3.5-7.8B-Instruct 사용
# 변경: EXAONE-3.5-2.4B-Instruct 사용 (경량 모델로 메모리 및 속도 최적화 목적)
# 변경일: 2025-06-10

import os
import json
import pandas as pd
from rdflib import Graph
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.schema import Document
import faiss
import torch
import re
import unicodedata
from konlpy.tag import Okt
import markdown
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. 경로 설정
base_path = "/mnt/e/chatbot_project_data/law_chatbot_dataset"
qa_data_root = os.path.join(base_path, "Law_Regulations_Text_Analysis_Data")
folders = {
    "terms_json": os.path.join(base_path, "law_knowledge_base", "법령용어"),
    "ontology_nt": os.path.join(base_path, "law_knowledge_base", "법률 데이터"),
    "feedback_log": os.path.join(base_path, "feedback_log.csv")
}

# 2. JSON / RDF 로딩 함수
def load_json_files(folder_path):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        data.append(json.load(f))
                except Exception as e:
                    print(f"[❌ JSON 로딩 실패] {file} → {e}")
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
                except:
                    pass
    return graphs

def extract_triples(graphs):
    triples = []
    for g in graphs:
        for s, p, o in g:
            triples.append({"subject": str(s), "predicate": str(p), "object": str(o)})
    return pd.DataFrame(triples)

# 3. QA 로딩
def extract_qa_from_clause_json(folder_path):
    qa_pairs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            label = str(data.get("unfavorableProvision") or "").strip()
                            if label:
                                qa_pairs.append({
                                    "question": "이 조항이 소비자에게 불리한가요?",
                                    "answer": label
                                })
                except:
                    pass
    return pd.DataFrame(qa_pairs)

def map_answer_label(label):
    if str(label) == "1":
        return "이 조항은 소비자에게 불리한 조항입니다."
    elif str(label) == "2":
        return "이 조항은 소비자에게 유리한 조항입니다."
    else:
        return "이 조항의 유불리는 명확하지 않습니다."

qa_pickle_path = os.path.join(base_path, "law_qa_df.pkl")
if os.path.exists(qa_pickle_path):
    print("✅ law_qa_df.pkl 로드 중...")
    law_qa_df = pd.read_pickle(qa_pickle_path)
else:
    print("⚠️ law_qa_df.pkl 없음 → JSON에서 추출합니다.")
    law_qa_df = extract_qa_from_clause_json(qa_data_root)
    law_qa_df["answer"] = law_qa_df["answer"].apply(map_answer_label)
    law_qa_df.to_pickle(qa_pickle_path)
    print("✅ 자동 저장 완료:", qa_pickle_path)

# 4. 벡터 DB
documents = [Document(page_content=f"{row['question']}\n{row['answer']}") for _, row in law_qa_df.iterrows()]
embedding_model = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
vectorstore = LangchainFAISS.from_documents(documents, embedding_model)

# 5. 재랭커
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 6. 용어 및 삼중항
terms_dict = {
    item["용어"]: item["정의"]
    for item in load_json_files(folders["terms_json"])
    if isinstance(item, dict) and "용어" in item and "정의" in item
}
law_triple_df = extract_triples(load_rdf_files(folders["ontology_nt"]))

# 7. EXAONE 모델 로딩 (4bit 최적화 + GPU 고정)
model_path = "./EXAONE-3.5-2.4B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

global_tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)

global_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="cuda"
)


def load_model_once():
    global global_tokenizer, global_model
    if global_tokenizer is None or global_model is None:
        print("🔄 EXAONE 모델 로컬에서 로딩 중...")
        global_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        global_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=False,
            quantization_config=bnb_config,
            device_map="cuda"
        )
        print("✅ 모델 디바이스:", global_model.device)
    return global_tokenizer, global_model


def ask_exaone(prompt):
    tokenizer, model = load_model_once()
    inputs = tokenizer(prompt[:1500], return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            repetition_penalty=1.1,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    end = time.time()
    print(f"⏱ 모델 응답 시간: {end - start:.2f}초")

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return markdown.markdown(result.replace(prompt.strip(), "").strip(), extensions=['markdown.extensions.tables'])

# 8. 유사 질문 검색
def retrieve_similar_qa(user_question, top_k=5):
    retrieved = vectorstore.similarity_search(user_question, k=top_k)
    qa_pairs = []
    for doc in retrieved:
        parts = doc.page_content.split("\n", 1)
        if len(parts) == 2:
            qa_pairs.append((parts[0], parts[1]))
    if not qa_pairs:
        return []
    scores = reranker.predict([[user_question, q] for q, _ in qa_pairs])
    return [pair for pair, _ in sorted(zip(qa_pairs, scores), key=lambda x: x[1], reverse=True)]

# 9. 보조 지식 응답
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

# 10. 전처리 및 키워드
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

# 11. 통합 챗봇 응답
def smart_legal_chat(user_input):
    term_def = lookup_legal_term_definition(user_input)
    if term_def:
        return term_def

    rdf_info = search_rdf_triple(user_input)
    if rdf_info:
        return f"🔎 RDF 정보:\n{rdf_info}"

    cleaned = clean_question(user_input)
    keywords = extract_keywords_morph(cleaned)
    top_qas = retrieve_similar_qa(cleaned, top_k=5)

    if top_qas:
        q, a = top_qas[0]
        prompt = f"""질문: {cleaned}
아래 내용을 참고하여 자연스럽게 답변해주세요:

Q: {q}
A: {a}

답변:"""
    else:
        prompt = f"질문: {cleaned}\n답변:"

    return ask_exaone(prompt)

# 12. 피드백 저장
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
