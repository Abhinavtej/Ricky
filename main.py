from flask import Flask, request, jsonify
from flask_cors import CORS
from generate import generate_screenplay
from retrieval import process_and_store

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "RAG Screenplay Assistant Backend Running!"})

@app.route("/generate", methods=["POST"])
def generate_splay():
    """Generate screenplay based on user input."""
    data = request.json
    text = data.get("text", "")
    genre = data.get("genre", "General")

    if not text:
        return jsonify({"error": "No input provided"}), 400

    screenplay = generate_screenplay(text, genre)
    return jsonify({"screenplay": screenplay})

@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload a file, extract text, generate embeddings, and upsert to Pinecone."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    genre = request.form.get("genre", "General")
    
    status = process_and_store(file, genre)
    return jsonify({"message": status})

if __name__ == "__main__":
    app.run(debug=True)
