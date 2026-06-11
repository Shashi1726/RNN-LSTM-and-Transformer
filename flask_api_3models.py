import json
import os
import re
import joblib
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from pyngrok import ngrok
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
PORT = 5000


GEMINI_API_KEY = "...."

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


MODEL_DIR = "/content/drive/MyDrive/goemotion_models"


NGROK_AUTH_TOKEN = "...."

PORT = 5000


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.fc(x)


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_all_models():
    print("Loading models from:", MODEL_DIR)

    config_path = os.path.join(MODEL_DIR, "config.json")
    labels_path = os.path.join(MODEL_DIR, "labels.pkl")

    config = load_json(config_path)
    labels = joblib.load(labels_path)

    logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    keras_tokenizer = joblib.load(os.path.join(MODEL_DIR, "tokenizer.pkl"))

    lstm_model = LSTMModel(
        vocab_size=int(config["vocab_size"]),
        embed_dim=128,
        hidden_dim=128,
        num_classes=int(config["num_classes"]),
    )
    lstm_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "lstm_model.pt"), map_location="cpu")
    )
    lstm_model.eval()

    transformer_path = os.path.join(MODEL_DIR, "transformer_model")
    transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
    transformer_model.eval()

    print("Models loaded successfully.")

    return {
        "config": config,
        "labels": labels,
        "logistic_model": logistic_model,
        "tfidf_vectorizer": tfidf_vectorizer,
        "keras_tokenizer": keras_tokenizer,
        "lstm_model": lstm_model,
        "transformer_tokenizer": transformer_tokenizer,
        "transformer_model": transformer_model,
    }


models = load_all_models()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(
        {
            "status": "ok",
            "message": "GoEmotions API running!",
            "models": ["logistic", "lstm", "transformer"],
        }
    )


@app.route("/models", methods=["GET"])
def available_models():
    return jsonify(
        {
            "models": ["logistic", "lstm", "transformer"],
            "default": "logistic",
        }
    )


@app.route("/labels", methods=["GET"])
def get_labels():
    return jsonify({"labels": models["labels"]})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()

    if not body or "text" not in body:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = body["text"]
    model_type = body.get("model", "logistic").lower()
    threshold = float(body.get("threshold", default_threshold(model_type)))

    try:
        if model_type == "logistic":
            emotions = run_logistic(text, threshold)
        elif model_type == "lstm":
            emotions = run_lstm(text, threshold)
        elif model_type in ["transformer", "bert", "distilbert"]:
            model_type = "transformer"
            emotions = run_transformer(text, threshold)
        else:
            return jsonify({"error": f"Unknown model '{model_type}'"}), 400
        
        coach_response = generate_emotion_coach_response(
            text,
            emotions
            )

        return jsonify(
            {
                "text": text,
                "model": model_type,
                "threshold": threshold,
                "count": len(emotions),
                "emotions": emotions,
                "coach_response": coach_response
            }
        )

    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


def default_threshold(model_type):
    if model_type == "logistic":
        return 0.15
    if model_type == "lstm":
        return 0.5
    return 0.5


def format_results(probs, threshold):
    results = []

    for i, probability in enumerate(probs):
        value = float(probability)
        if value > threshold:
            results.append(
                {
                    "emotion": models["labels"][i],
                    "probability": round(value, 4),
                }
            )

    results.sort(key=lambda item: -item["probability"])
    return results

def generate_emotion_coach_response(text, emotions):

    if not emotions:
        return (
            "I couldn't identify a strong emotional pattern. "
            "Can you tell me a little more about how you're feeling?"
        )

    emotion_summary = ", ".join(
        [
            f"{e['emotion']} ({e['probability']:.2f})"
            for e in emotions[:5]
        ]
    )

    prompt = f"""
You are an emotionally intelligent AI assistant.

User Message:
{text}

Detected Emotions:
{emotion_summary}

Your task:

1. Explain what the user may be feeling.
2. Be empathetic.
3. Give practical advice.
4. Keep response under 120 words.
5. Do NOT diagnose mental illnesses.
6. Speak directly to the user.

Response:
"""

    try:
        response = gemini_model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"Gemini Error: {str(e)}"

def run_logistic(text, threshold):
    cleaned = preprocess(text)
    vec = models["tfidf_vectorizer"].transform([cleaned])
    probs = models["logistic_model"].predict_proba(vec)[0]
    return format_results(probs, threshold)


def run_lstm(text, threshold):
    cleaned = preprocess(text)
    seq = models["keras_tokenizer"].texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=int(models["config"]["max_len"]),
        padding="post",
    )
    tensor = torch.tensor(padded, dtype=torch.long)

    with torch.no_grad():
        outputs = models["lstm_model"](tensor)
        probs = torch.sigmoid(outputs).squeeze().numpy()

    return format_results(probs, threshold)


def run_transformer(text, threshold):
    cleaned = preprocess(text)
    tokenizer = models["transformer_tokenizer"]
    transformer_model = models["transformer_model"]

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()

    return format_results(probs, threshold)


def start_ngrok():
    if NGROK_AUTH_TOKEN.strip():
        ngrok.set_auth_token(NGROK_AUTH_TOKEN.strip())

    ngrok.kill()
    tunnel = ngrok.connect(PORT)
    public_url = tunnel.public_url

    print("\n" + "=" * 58)
    print("  GoEmotions API is LIVE")
    print("=" * 58)
    print(f"  URL      ->  {public_url}")
    print("\n  Paste this URL into your website API URL box")
    print("\n  Endpoints:")
    print(f"    GET  {public_url}/ping")
    print(f"    GET  {public_url}/models")
    print(f"    GET  {public_url}/labels")
    print(f"    POST {public_url}/predict")
    print("\n  Example body:")
    print('    {"text": "I am happy", "model": "logistic"}')
    print('    {"text": "I am happy", "model": "lstm"}')
    print('    {"text": "I am happy", "model": "transformer"}')
    print("=" * 58, flush=True)

    app.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False)


if __name__ == "__main__":
    start_ngrok()


