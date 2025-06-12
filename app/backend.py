from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import importlib.util
from sentence_transformers import CrossEncoder
import csv
import traceback
import sys

# ✅ transformers_modules 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
custom_module_path = os.path.join(parent_dir, "transformers_modules")
if custom_module_path not in sys.path:
    sys.path.append(custom_module_path)

# 허깅 페이스스
import logging
logging.basicConfig(level=logging.DEBUG)

# 🔹 전처리 함수 불러오기
from app.utils import preprocessing

# Flask 앱 생성
app = Flask(__name__, template_folder="../templates")

# 파일 업로드 디렉토리 설정
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 이름과 해당 Python 파일 매핑
model_map = {
    "EXAONE3.5_2.4B": "models/EXAONE3.5_2.4B.py",
    "kogpt2-base-v2": "models/kogpt2-base-v2.py",
    "polyglot_1.3b": "models/polyglot_1.3b.py"
}

# 🔹 Reranker 모델 불러오기
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 🔹 메인 페이지 요청 시 HTML 반환
@app.route("/")
def serve_html():
    return render_template("main.html")

# 🔹 질문 처리
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "")
    model_name = request.form.get("model", "")
    uploaded_file = request.files.get("file")

    if not question:
        return jsonify({"answer": "질문이 비어 있습니다."}), 400

    if model_name not in model_map:
        return jsonify({"answer": f"지원하지 않는 모델입니다: {model_name}"}), 400

    model_file = model_map[model_name]
    if not os.path.exists(model_file):
        return jsonify({"answer": f"모델 파일을 찾을 수 없습니다: {model_file}"}), 404

    file_path = ""
    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)\

    try:
        # ✅ 형태소 기반 키워드 추출
        clean_q = preprocessing.clean_question(question)
        morph_keywords = preprocessing.extract_keywords_morph(clean_q)
        keyword_info = f"[🔍 주요 키워드]: {', '.join(morph_keywords)}"

        # 모델 모듈 로딩
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        if hasattr(model_module, "smart_legal_chat"):
            results = model_module.search_similar_questions(question, top_k=5)
            if results:
                rerank_input = [[question, r[0]] for r in results]
                scores = reranker.predict(rerank_input)
                best_idx = scores.argmax()
                best_question, best_answer = results[best_idx]

                prompt = f"{keyword_info}\n\n사용자 질문: {question}\n\n관련된 기존 질문: {best_question}\n\n기존 답변: {best_answer}\n\n이 내용을 참고하여 사용자 질문에 대해 자세히 설명해주세요."
                answer = model_module.ask_exaone(prompt)
            else:
                answer = model_module.smart_legal_chat(question)

        elif hasattr(model_module, "run_model"):
            answer = model_module.run_model(question, file_path)
        else:
            return jsonify({"answer": f"모델에 실행 가능한 함수가 없습니다."}), 500

        return jsonify({"answer": answer})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"모델 실행 중 오류 발생: {str(e)}"}), 500

# 🔹 피드백 저장
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    question = data.get("question", "")
    model = data.get("model", "")
    answer = data.get("answer", "")
    feedback = data.get("feedback", "")

    feedback_file = "feedback_log.csv"
    is_new_file = not os.path.exists(feedback_file)

    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["model", "question", "answer", "feedback"])
        writer.writerow([model, question, answer, feedback])

    return jsonify({"status": "success"})

# 🔹 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
