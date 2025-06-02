from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import importlib.util

# Flask 앱 생성
app = Flask(__name__)

# 파일 업로드 디렉토리 설정
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 폴더가 없으면 생성

# 모델 이름과 해당 Python 파일 매핑
model_map = {
    "EXAONE3.5_7.8B": "models/EXAONE3.5_7.8B.ipynb",
    "KLUE_RoBERTa": "models/KLUE_RoBERTa.ipynb",
    "KoreALBERT": "models/KoreALBERT.ipynb",
}

# 🔹 메인 페이지 요청 시 HTML 반환
@app.route("/")
def serve_html():
    return send_from_directory(".", "tax_chatbot.html")  # 현재 디렉토리에서 HTML 파일 반환

# 🔹 POST 요청 처리 (질문, 모델, 파일 받기)
@app.route("/ask", methods=["POST"])
def ask():
    # 폼 데이터 가져오기
    question = request.form.get("question", "")
    model_name = request.form.get("model", "")
    uploaded_file = request.files.get("file")

    # 모델 이름 유효성 검사
    if model_name not in model_map:
        return jsonify({"answer": f"지원하지 않는 모델입니다: {model_name}"}), 400

    model_file = model_map[model_name]

    # 모델 파일 존재 여부 확인
    if not os.path.exists(model_file):
        return jsonify({"answer": f"모델 파일이 없습니다: {model_file}"}), 404

    # 업로드된 파일 저장
    file_path = ""
    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)  # 보안 처리된 파일명
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(file_path)

    try:
        # 모델 파일 불러오기 (모듈 동적 로딩)
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # run_model 함수 실행
        answer = model_module.run_model(question, file_path)

        # 응답 반환
        return jsonify({"answer": answer})

    except Exception as e:
        # 예외 발생 시 에러 메시지 반환
        return jsonify({"answer": f"모델 실행 오류: {str(e)}"}), 500

# 🔹 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # 외부에서 접근 가능, 디버그 모드 ON