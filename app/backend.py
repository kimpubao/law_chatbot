from flask import Flask, request, jsonify, render_template 
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
    "EXAONE3.5_7.8B": "models/EXAONE3.5_7.8B.py",
    "KLUE_RoBERTa": "models/KLUE_RoBERTa.py",
    "KoreALBERT": "models/KoreALBERT.py"
}

# 🔹 메인 페이지 요청 시 HTML 반환
@app.route("/")
def serve_html():
    return render_template("main.html") # templates 폴더에서 main.html 자동 로드

# 🔹 POST 요청 처리 (질문, 모델, 파일 받기)
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
        uploaded_file.save(file_path)

    try:
        # 모델 파일 동적 로딩
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # run_model 또는 smart_legal_chat 함수 호출
        if hasattr(model_module, "run_model"):
            answer = model_module.run_model(question, file_path)
        elif hasattr(model_module, "smart_legal_chat"):
            answer = model_module.smart_legal_chat(question)
        else:
            return jsonify({"answer": f"모델에 실행 가능한 함수가 없습니다."}), 500

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"모델 실행 중 오류 발생: {str(e)}"}), 500

# 🔹 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)