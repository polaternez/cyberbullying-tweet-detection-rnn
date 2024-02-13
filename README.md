# Cyberbullying Tweet Detector 2: Project Overview  
This tool flags potential cyberbullying tweets by classifying them by type (not_cyberbullying, gender, religion, other_cyberbullying, age, ethnicity).

* Take Cyberbullying Classification Dataset from Kaggle
* Cleaning the data
* Apply data preprocessing steps to cleaned data
* Build Recurrent Neural Network(RNN) using Long Short-Term Memory(LSTM) layers, then evaluate them on test dataset
* Built a client facing API using Flask 

Note: This project was made for educational purposes.

## Technologies and Resources
* **Python Version:** 3.10  
* **Libraries:** numpy, pandas, matplotlib, seaborn, nltk, tensorflow, sklearn, flask, json  
* **Flask API Setup:**
  * ```pip install -r requirements.txt```  
  *  ```conda env create -n <ENVNAME> -f environment.yaml```  (Anaconda environment)
   
* **Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## Getting Data
We use the <a href="https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification">Cyberbullying Classification</a> dataset from Kaggle. This dataset contains more than 47000 tweets labelled according to the class of cyberbullying:
* Not Cyberbullying
* Gender
* Religion
* Other type of cyberbullying
* Age
* Ethnicity

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")


## Data Cleaning
We create a python script to clear text data, its apply the following operations to the text:
* Removing Puncuatations
* Removing Numbers
* Lowecasing the data
* Remove stop words
* Lemmatize/ Stem words
* Remove URLs

## Data Preprocessing
We apply the TextVectorization layer from Keras to previously cleaned tweets. This layer one-hot encodes text, returning a list of encoded integers, each corresponding to a word (or token) in the given input string, and then pads sequences to the same length.


## Model Building 

First, we split the data into train and test sets with a test size of 20%. After that, we build following RNN using Bidirectional LSTM layers:

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/model.png "LSTM Model")

We measure the model loss with "categorical_crossentropy" and optimize the model with "RMSprop". After training, we get the following results:

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/results.jpg "Model Performances")

## Productionization 
In this step, we created the UI with the Flask. API endpoint help receives a request tweets and returns the results of the cyberbullying type prediction.

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/flask-api.png "Cyberbullying Tweet Detector 2")






