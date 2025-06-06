# 🧑‍⚖️ Korean Legal Chatbot using RAG (Retrieval-Augmented Generation)

이 프로젝트는 한국어 법률 데이터를 기반으로 RAG(Retrieval-Augmented Generation) 구조를 활용하여 구축한 법률 특화 챗봇 시스템입니다. 총 3종의 LLM 기반 모델을 활용하여 성능 비교 및 최적화를 수행하며, 실제 법률 QA 시스템 개발에 필요한 기술 스택을 통합적으로 구성하였습니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 프로젝트 명 | Korean Legal RAG Chatbot |
| 프로젝트 시작일 | 2025년 5월 27일 (화요일) |
| 목적 | 한국 법률 정보 기반 질의응답 제공 시스템 구현 |
| 모델 | EXAONE (생성형 LLM), KLUE-RoBERTa / KoreALBERT (추론형 모델) |
| 검색기 | KR-SBERT 기반 벡터 검색 (FAISS 등 연동 예정) |
| 데이터 출처 | AI Hub (법률 QA, 판례, 규정 문서 등) |
| 주요 기능 | 질의 기반 답변 생성, 정확한 문헌 인용, 다양한 모델 성능 비교 |
| 대상 사용자 | 일반 사용자, 법률 초심자, 기업 내부 시스템 등 |

---

## 🛠️ 기술 스택 및 구현 환경

- 운영체제 
    
    ![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-E95420?logo=ubuntu) ![WSL](https://img.shields.io/badge/WSL-2-green?logo=windows)

- 언어 

    ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
 
- 백엔드 프레임워크

  ![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C?logo=pytorch)  
  ![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface)  
  ![Accelerate](https://img.shields.io/badge/Accelerate-Inference-lightgrey)

- 임베딩/검색기

  ![SBERT](https://img.shields.io/badge/SentenceBERT-Embedding-blueviolet)  
  ![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-critical)

- 프론트엔드

    ![HTML](https://img.shields.io/badge/HTML-UI-orange?logo=html5)

- 개발 도구

  ![VSCode](https://img.shields.io/badge/VSCode-IDE-blue?logo=visualstudiocode)  
  ![WSL](https://img.shields.io/badge/WSL-2-green?logo=windows)

- RAG 구성요소
  - Retriever: SBERT 기반 벡터 검색  
  - Generator: 사전 학습 LLM 3종 비교 (EXAONE / KLUE-RoBERTa / KoreALBERT)

---
## 시스템 구성도 (RAG 구조)
![RAG Architecture](./static/rag_architecture.png)
---

## 🤖 사용 모델 소개 및 논문
### EXAONE 3.5 7.8B  
- Hugging Face: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-instruct  
- 논문: https://arxiv.org/abs/2412.04862
### KLUE-RoBERTa-base  
- Hugging Face: https://huggingface.co/klue/roberta-base  
- 논문: https://arxiv.org/abs/2105.09680

### KoreALBERT (KcELECTRA-base)  
- Hugging Face: https://huggingface.co/beomi/KcELECTRA-base  
- 논문: https://arxiv.org/abs/2004.13922

---

## 🗃️ 활용 데이터셋 (AI Hub)
## 데이터셋 구성 및 활용 전략
- 본 프로젝트는 각기 다른 AI Hub 공개 데이터셋 3종을 활용하여, Retrieval-Augmented Generation (RAG) 기반 법률 챗봇을 설계하였습니다.
- 각 데이터셋은 챗봇 시스템의 다른 기능 모듈에 맞추어 전략적으로 분리 활용됩니다.

### 구성 전략
1. 질문-답변 형식 데이터 (AIHub 데이터셋 ①)
  → 사용자 자연어 질문에 대한 초기 응답 생성 구조 구성에 활용
  → LLM 기반 응답의 학습/튜닝 또는 예시 제공용

2. 법령/판결문/약관 텍스트 (AIHub 데이터셋 ②)
  → 정확한 조문/판결문 인용 응답을 위한 RAG Retriever 문서베이스 구성
  → 문서를 Chunking하여 긴 문맥 처리 기반의 정밀 응답 보완

3. 법률 지식 그래프용 관계 데이터 (AIHub 데이터셋 ③)
  → 내부 지식그래프 구축 또는 키워드 기반 추천 기능에 활용
  → 향후 유사 질의 추천, 질문 카테고리 자동 분류 등에 확장 가능

## 데이터셋별 전략적 활용 요약

|데이터셋 출처|역할|활용 방식|
|---|---|---|
|① 질의응답 형식 데이터|사용자 질문-응답 구조 구성|예시 기반 생성형 학습 or 평가
|② 판결/법령/약관 텍스트|근거 문서 기반 응답|문서 Chunk + RAG Retrieval
|③ 관계형 데이터 (그래프)|키워드 연관, 추천 기능|지식그래프 또는 분류 라벨링

## 데이터 출처
| 이름 | 설명 | URL |
|------|------|------|
| 법률 QA 데이터 | 질의-응답 쌍으로 구성된 지식 기반 | https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=99 |
| 법령·판결문 텍스트 | 정확한 문서 기반 응답 생성을 위한 데이터 | https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=580 |
| 법률 지식 그래프용 데이터 | 문서 분류 및 필터링용 관계 데이터 | https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71722 |

---
## 📁 데이터 폴더 구조
```
chatbot_project_data/
└── law_chatbot_dataset/
    ├── law_knowledge_base/
    │   └── 질문-답변 형식 데이터셋
    ├── law_Knowledge_Based_Relationship_Data/
    │   └── 관계 기반 지식 그래프 데이터
    └── law_Regulations_Text_Analysis_Data/
        └── 법령, 판결문, 약관 텍스트 데이터
```

## 📁 프로젝트 디렉토리 구조
```
law_chatbot/
├── app/
│ ├── backend.py ← Flask 서버
│ ├── main_ui.py ← Gradio UI 실행기
│ ├── __init__.py
│ └── utils/ ← 전처리, OCR 등 공통 함수
│     └── preprocessing.py
│     └── __init__.py
│
├── models/ ← 모델별 실행 로직 분리
│ ├── model_KoAlpaca7B.py
│ ├── model_KoAlpaca12B.py
│ ├── model_Llama3.py
│ └── model_Mistral.py
│
├── templates/ ← HTML 기반 UI (Flask용)
│   └── main.html
│
├── static/ ← 구조도, 출력 이미지 등 정적 파일
│   └── RAG_architecture.png
│
├── uploads/ ← (예시 전용)사용자 업로드 파일 저장소
│
├── data/ ← (예시 전용) 테스트용 데이터셋
│   └── sample_dataset.json ← 실제 대용량은 외장하드에 있음
│
├── README.md
├── requirements.txt
└── .gitignore

```

---
## 📊 모델별 성능 비교

| 모델                     | 파라미터 수 | 응답 정확도           | 응답 속도         | 메모리 사용량 | 비고           |
| ---------------------- | ------ | ---------------- | ------------- | ------- | ------------ |
| EXAONE 3.5 7.8B        | 약 7.8B | 높음 *(경험 기반)*     | 느림 *(테스트 기준)* | 높음      | 지시문 최적화      |
| KLUE-RoBERTa           | 약 110M | 중간 *(공식 보고서 기준)* | 빠름            | 낮음      | 분류형 QA 적합    |
| KoreALBERT (KcELECTRA) | 약 13M  | 낮음\~중간 *(경량 모델)* | 매우 빠름         | 매우 낮음   | 모바일/경량 환경 적합 |


---

## 💬 실행 예시 (EXAONE 생성 응답 기준 예시)

**입력**  
  - 한국에서 사기죄가 성립되기 위한 요건은 무엇인가요?

**출력 (EXAONE 기반 예상 답변 / 추후 실제 답변으로 수정정)** 
- 사기죄는 행위자가 타인을 기망하여 착오에 빠뜨리고, 그로 인해 재산상 이익을 취득하거나 손해를 발생시키는 경우 성립합니다. 이는 형법 제347조에 명시되어 있습니다.
---

## 🗓️ 개발 일정 및 이력
| 날짜 | 주요 작업 |
|------|-----------|
| 2025.05.27 | 데이터 수집 및 프로젝트 착수 |
| 2025.05.29~ | 모델 구조 구현 및 토큰 테스트 |
| 2025.06.23~ | 각 모델별 응답 비교 실험, 문서화 진행 |
| 예정 | FAISS 벡터 검색기 연동, UI 완성, 리포트 생성 |

## 📦 설치 방법 (Installation)

Python 가상환경을 설정한 후 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```
---
## 🔧 실행 방법

Flask 백엔드 서버를 실행하기 위해 아래 명령어를 순서대로 입력합니다:

```bash
# 현재 디렉토리를 PYTHONPATH에 추가 (모듈 import 오류 방지용)
export PYTHONPATH=.

# 백엔드 서버 실행
python app/backend.py
```
## 요약 비교
| 방식                         | 장점                    | 단점                               |
| -------------------------- | --------------------- | -------------------------------- |
| `python app/backend.py`    | 간단함                   | `PYTHONPATH` 설정 없으면 import 에러 가능 |
| `export PYTHONPATH=.` 후 실행 | 패키지 import 경로 확실히 설정됨 | 한 번 더 명령어 입력해야 함                 |
---

## 🎯 포트폴리오 활용 포인트

- 한국어 기반 RAG 구조 직접 구현 경험
- 대형 LLM + 벡터 검색기 통합 구현
- Hugging Face 기반 실전 모델 사용 및 비교 분석
- 실제 법률 QA 데이터셋을 활용한 의미 있는 실험
- 학습, 추론, 리소스 최적화 등 실무 기반 문제 해결 능력

---

## 🚀 향후 발전 방향
- 예정: 성능 비교 리포트 자동 생성 시스템 구축
- 예정: Reranker + OCR 기반 문서 필터링 정확도 향상
- 예정: 판례 요약 + 조문 연결 + 인용문 추천 기능 개발
- 예정: Gradio UI 또는 HTML 프론트엔드와의 통합

---

![Author](https://img.shields.io/badge/Author-김상준-blue)

![GitHub](https://img.shields.io/badge/GitHub-www.github.com/kimpubao-black?logo=github)

![Email](https://img.shields.io/badge/Email-dfg7785@gmail.com-red?ogo=gmail)