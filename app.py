from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained LSTM model
model = load_model("LSTM_HateSpeech_Model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to predict toxicity
def predict_toxicity(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen to match training
    prediction = model.predict(padded_sequence)[0][0]
    return "Toxic" if prediction > 0.5 else "Non-Toxic"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    result = predict_toxicity(text)
    return jsonify({"text": text, "prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
