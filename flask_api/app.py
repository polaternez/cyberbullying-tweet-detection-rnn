import json
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

import text_cleaning as tc


VOCAB_SIZE = 50000  # vocabulary size
MAX_LEN = 200

# cyberbullying types
LABEL_MAPPING = {
    0: 'not_cyberbullying',
    1: 'gender',
    2: 'religion',
    3: 'other_cyberbullying',
    4: 'age',
    5: 'ethnicity'
}

# Load and compile model
MODEL_PATH = "models/cyberbullying_lstm.h5"
model = load_model(MODEL_PATH,  compile=False)
model.compile(optimizer=tf.optimizers.RMSprop(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Prepare vectorization layer
def prepare_vectorization_layer():
    df = pd.read_csv("../data/cleaned_tweets.csv")
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

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        tweet_text = request.form["tweet_text"]
        cleaned_text = tc.clean_data(tweet_text)
        vectorized_text = vectorize_layer([cleaned_text])
        
        # predictions
        prediction = model.predict(vectorized_text.numpy())
        predicted_label = LABEL_MAPPING[np.argmax(prediction)]

        return render_template("index.html", prediction_text=predicted_label, tweet=f"<< {tweet_text} >>")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

@app.route("/predict_api", methods=["POST"])
def predict_api():
    '''
    For direct API calls trought request
    '''
    try:
        data = request.get_json()
        if not data or "input" not in data:
            return json.dumps({"error": "Invalid input"}), 400
        
        df = pd.DataFrame(data)
        cleaned_text = df["input"].apply(tc.clean_data)
        vectorized_text = vectorize_layer(cleaned_text)

        # predictions
        prediction = model.predict(vectorized_text.numpy())
        predicted_labels = [LABEL_MAPPING[np.argmax(pred)] for pred in prediction]
        
        return json.dumps({'response': predicted_labels}), 200
    except Exception as e:
        return json.dumps({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


