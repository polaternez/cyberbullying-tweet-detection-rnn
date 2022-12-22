# Cyberbullying Tweet Detector 2: Project Overview  
* This tool was created to cyberbullying tweet type(not_cyberbullying, gender, religion, other_cyberbullying, age, ethnicity) prediction
* Take Cyberbullying Classification Dataset from Kaggle
* Cleaning the data
* Apply data preprocessing steps to cleaned data
* Build LSTM(Long Short-Term Memory) model,then evaluate them on test dataset
* Built a client facing API using Flask 

Note: This project was made for educational purposes.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** numpy, pandas, matplotlib, seaborn, nltk, tensorflow, sklearn, flask, json  
**For Flask API Requirements:**  ```pip install -r requirements.txt```  
**For Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## Getting Data
We use the <a href="https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification">Cyberbullying Classification</a> dataset from Kaggle. This dataset contains more than 47000 tweets labelled according to the class of cyberbullying:
* Not Cyberbullying
* Gender
* Religion
* Other type of cyberbullying
* Age
* Ethnicity

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj_v2/blob/documentation/images/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")


## Data Cleaning
We create a python script to clear text data, its apply the following operations to the text:
* Removing Puncuatations
* Removing Numbers
* Lowecasing the data
* Remove stop words
* Lemmatize/ Stem words
* Remove URLs


## Model Building 

First, we apply TextVectorization to clean tweets for one-hot encoding and padding them. Then split the data into train and test sets with a test size of 20%. After that, We build following LSTM(Long Short-Term Memory) model:

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj_v2/blob/documentation/images/model.png "LSTM Model")

We measure the model loss with "categorical_crossentropy" and optimize the model with "RMSprop". After training, we get the following results:

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj_v2/blob/documentation/images/results.jpg "Model Performances")

## Productionization 
In this step, we created the UI with the Flask. API endpoint help receives a request tweets and returns the results of the cyberbullying type prediction.

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj_v2/blob/documentation/images/flask-api.png "Cyberbullying Tweet Detector 2")






