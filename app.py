import os
from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

app = Flask(__name__)

# Muat model dan tokenizer
model_path = 'bert_model'  # Path ke model yang sudah disimpan
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

# Fungsi untuk melakukan prediksi sentimen
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    logits = model(**inputs).logits
    predicted_class = tf.argmax(logits, axis=-1).numpy()[0]
    return "Positive" if predicted_class == 1 else "Negative"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    if request.method == "POST":
        text = request.form["text"]
        sentiment = predict_sentiment(text)
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
