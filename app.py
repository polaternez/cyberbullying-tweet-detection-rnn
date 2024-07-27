import json
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import layers
from keras.models import load_model

import text_cleaning as tc


VOCAB_SIZE = 50000  # vocabulary size
MAX_LEN = 200

# cyberbullying types
label_dict = {
    0: 'not_cyberbullying',
    1: 'gender',
    2: 'religion',
    3: 'other_cyberbullying',
    4: 'age',
    5: 'ethnicity'
}

# Load the model
model = load_model("models/cyberbullying_lstm.h5",  compile=False)
model.compile(optimizer=tf.optimizers.RMSprop(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Adapt vectorization layer
cleaned_tweets_df = pd.read_csv("data/cleaned_tweets.csv")
cleaned_tweets_df.dropna(inplace=True)

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize='lower',
    output_mode='int',
    output_sequence_length=MAX_LEN
)
vectorize_layer.adapt(cleaned_tweets_df["clean_text"])


# WSGI
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    new_data = request.form["tweet_text"]
    temp_df = pd.DataFrame(np.array(new_data).reshape(-1, 1), columns=['tweet_text'])
    cleaned_text = temp_df["tweet_text"].apply(tc.clean_data)
    vectorized_text = vectorize_layer(cleaned_text)
    
    # predictions
    prediction = model.predict(vectorized_text.numpy())
    
    # output
    output = np.argmax(prediction, axis=1)
    output_text = label_dict[int(output)]
    return render_template("index.html", prediction_text=output_text, tweet=f"<< {new_data} >>")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    '''
    For direct API calls trought request
    '''
    request_json = request.get_json()
    temp_df = pd.DataFrame(request_json)
    cleaned_text = temp_df["input"].apply(tc.clean_data)
    vectorized_text = vectorize_layer(cleaned_text)

    # predictions
    prediction = model.predict(vectorized_text.numpy())
    
    # output
    output = np.argmax(prediction, axis=1)
    output_text = [label_dict[x] for x in output]
    response = json.dumps({'response': output_text})
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)


