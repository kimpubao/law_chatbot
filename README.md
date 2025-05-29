# 🧑‍⚖️ Korean Legal Chatbot using RAG (Retrieval-Augmented Generation)

이 프로젝트는 한국어 법률 데이터를 기반으로 RAG(Retrieval-Augmented Generation) 구조를 활용하여 구축한 법률 특화 챗봇 시스템입니다. 총 3종의 LLM 기반 모델을 활용하여 성능 비교 및 최적화를 수행하며, 실제 법률 QA 시스템 개발에 필요한 기술 스택을 통합적으로 구성하였습니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 프로젝트 명 | Korean Legal RAG Chatbot |
| 프로젝트 시작일 | 2025년 5월 27일 (화요일) |
| 목적 | 한국 법률 정보 기반 질의응답 제공 시스템 구현 |
| 모델 | EXAONE 3.5 7.8B / KLUE-RoBERTa / KoreALBERT |
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

## 📊 모델별 성능 비교 (예시)

| 모델 | 파라미터 수 | 응답 정확도 | 응답 속도 | 메모리 사용량 | 비고 |
|-------|--------------|----------------|-------------|----------------|------|
| EXAONE 3.5 7.8B | 약 7.8B | 높음 | 느림 | 높음 | 지시문 최적화, 고성능 |
| KLUE-RoBERTa | 약 110M | 중간 | 빠름 | 낮음 | 분류/추론 기반 QA 적합 |
| KoreALBERT (KcELECTRA) | 약 13M | 낮음~중간 | 매우 빠름 | 매우 낮음 | 경량 환경에 적합 |

---

## 💬 실행 예시 (샘플 입력/출력)

**입력**  
    - 한국에서 사기죄가 성립되기 위한 요건은 무엇인가요?

**출력 (EXAONE 기반)** 
- 사기죄가 성립하기 위해서는 행위자가 타인을 기망하여 착오에 빠뜨리고, 그로 인해 재산상 이익을 취하거나 손해를 가한 경우에 성립합니다. (형법 제347조 참조)
---

## 🗓️ 개발 일정 및 이력
| 날짜 | 주요 작업 |
|------|-----------|
| 2025.05.27 | 데이터 수집 및 프로젝트 착수 |
| 2025.05.29~ | 모델 구조 구현 및 토큰 테스트 |
| 2025.06.20~ | 각 모델별 응답 비교 실험, 문서화 진행 |
| 예정 | FAISS 벡터 검색기 연동, UI 완성, 리포트 생성 |

---

## 🎯 포트폴리오 활용 포인트

- 한국어 기반 RAG 구조 직접 구현 경험
- 대형 LLM + 벡터 검색기 통합 구현
- Hugging Face 기반 실전 모델 사용 및 비교 분석
- 실제 법률 QA 데이터셋을 활용한 의미 있는 실험
- 학습, 추론, 리소스 최적화 등 실무 기반 문제 해결 능력

---

## 🚀 향후 발전 방향

- 성능 비교 리포트 자동 생성 시스템 구축
- Reranker + OCR 기반 문서 필터링 정확도 향상
- 판례 요약 + 조문 연결 + 인용문 추천 기능 개발
- Gradio UI 또는 HTML 프론트엔드와 연동

---

![Author](https://img.shields.io/badge/Author-김상준-blue)

![GitHub](https://img.shields.io/badge/GitHub-www.github.com/kimpubao-black?logo=github)

![Email](https://img.shields.io/badge/Email-dfg7785@gmail.com-red?ogo=gmail)