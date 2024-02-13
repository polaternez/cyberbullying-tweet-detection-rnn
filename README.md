# Cyberbullying Tweet Detector 2: Project Overview  
This project aims to develop a tool for identifying cyberbullying tweets and classifying them based on various categories such as gender, religion, age, ethnicity, and other types of cyberbullying. The primary objectives include:

- Utilizing the Cyberbullying Classification Dataset sourced from Kaggle.
- Conducting data cleaning procedures to enhance data quality.
- Applying data preprocessing techniques to prepare the cleaned data for analysis.
- Constructing a Recurrent Neural Network (RNN) model using Long Short-Term Memory (LSTM) layers and evaluating its performance on a separate test dataset.
- Implementing a client-facing API using Flask for seamless integration and usability.

## Technologies and Resources
* **Python Version:** 3.10  
* **Libraries:** numpy, pandas, matplotlib, seaborn, nltk, tensorflow, sklearn, flask, json  
* **Flask API Setup:**
  * ```pip install -r requirements.txt```  
  *  ```conda env create -n <ENVNAME> -f environment.yaml```  (Anaconda environment)
   
* **Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## Data Acquisition
The project relies on the [Cyberbullying Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) obtained from Kaggle. This dataset comprises over 47,000 labeled tweets categorized into distinct classes of cyberbullying.

- Not Cyberbullying
- Gender
- Religion
- Other types of cyberbullying
- Age
- Ethnicity

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")


## Data Cleaning
A custom Python script is developed to perform rigorous data cleaning processes. These processes involve:

- Removal of punctuation marks
- Elimination of numerical characters
- Conversion of text to lowercase
- Elimination of stop words
- Lemmatization/Stemming of words
- Removal of URLs

## Data Preprocessing
To prepare the cleaned tweets for analysis, the TextVectorization layer from Keras is applied. This layer facilitates one-hot encoding of text, resulting in a list of encoded integers representing individual words (or tokens) in the input string. Additionally, sequences are padded to ensure uniform length.

## Model Building 
1. **Train-Test Split:** Data is divided into 80% training and 20% testing sets.
2. **Bidirectional LSTM Model:** Build an RNN architecture utilizing Bidirectional LSTM layers.

**Model Visualization:**

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/model.png "LSTM Model")

3. **Evaluation:** We employ "categorical_crossentropy" for loss measurement and "RMSprop" for optimization.

**Model Performance:**

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/results.jpg "Model Performances")

## Productionization 
A Flask-based user interface (UI) allows users to submit tweets and receive cyberbullying type predictions in real-time.

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection-rnn/blob/master/reports/figures/flask-api.png "Cyberbullying Tweet Detector 2")






