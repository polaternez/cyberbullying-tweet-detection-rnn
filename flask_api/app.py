import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from text_cleaning import TextCleaner


VOCAB_SIZE = 50000  # vocabulary size
MAX_LEN = 200

LABEL_MAPPING = {
    0: 'not_cyberbullying',
    1: 'gender',
    2: 'religion',
    3: 'other_cyberbullying',
    4: 'age',
    5: 'ethnicity'
}

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "models/cyberbullying_lstm.keras"
model = load_model(MODEL_PATH)

# Handles text cleaning
cleaner = TextCleaner()

# Prepare vectorization layer
def prepare_vectorization_layer():
    df = pd.read_csv("../data/tweets_clean.csv")
    df.dropna(inplace=True)

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        standardize='lower',
        output_mode='int',
        output_sequence_length=MAX_LEN
    )
    vectorize_layer.adapt(df["clean_text"])
    return vectorize_layer

vectorize_layer = prepare_vectorization_layer()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        tweet_text = request.form["tweet_text"]
        cleaned_text = cleaner.clean(tweet_text)
        vectorized_text = vectorize_layer([cleaned_text])
        
        # Predictions
        prediction = model.predict(vectorized_text.numpy())
        predicted_label = LABEL_MAPPING[np.argmax(prediction)]

        return render_template(
            "index.html", 
            prediction_text=predicted_label, 
            tweet=f"<< {tweet_text} >>"
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    '''
    Handles direct API calls via JSON payload
    '''
    try:
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Invalid input layout. Expected JSON key 'input'"}), 400
        
        df = pd.DataFrame(data)
        cleaned_text = df["input"].apply(cleaner.clean)
        vectorized_text = vectorize_layer(cleaned_text)

        # Predictions
        prediction = model.predict(vectorized_text.numpy())
        predicted_labels = [LABEL_MAPPING[np.argmax(pred)] for pred in prediction]
        
        return jsonify({'response': predicted_labels}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)


