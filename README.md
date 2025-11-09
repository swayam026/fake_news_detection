Project deployed on streamlit : https://real-or-fake-news.streamlit.app/

Fake News Detection using NLP and Machine Learning

1. Project Description

This project focuses on detecting whether a news article or headline is real or fake using Natural Language Processing (NLP) and Machine Learning techniques.
It aims to classify text data into two categories — Real or Fake — based on linguistic and stylistic patterns in the content.
The model is trained using the TF-IDF (Term Frequency–Inverse Document Frequency) method for feature extraction and a Logistic Regression classifier for prediction.
The trained model is then deployed through a Streamlit web application, allowing users to input any news text and receive a real-time prediction with confidence scores.

2. Dataset Information

Dataset Used: ISOT Fake News Dataset
Dataset Link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Description:
The ISOT Fake News Dataset contains news articles labeled as real or fake, collected between 2016 and 2017.
The real news articles were scraped from Reuters.com, a trusted international news source.
The fake news articles were collected from unreliable sources flagged by PolitiFact and Wikipedia.

Each article includes:
title – The headline of the news article
text – The full content of the news article
subject – The category of the news (e.g., politics, world news)
date – The publication date

The dataset primarily covers political and world news topics and contains approximately 12,600 fake and 12,600 real articles, making it balanced for binary classification.



3. Folder Structure

The main project directory should contain the following files and folders.
Create a folder named data in the main directory and place both True.csv and Fake.csv dataset files inside it.
All other project files such as app.py, model.pkl, confusion_matrix.png, requirements.txt, and README.md should remain in the main directory.

Your project should look like this:

The data folder contains: 
True.csv – the dataset file containing real news articles
Fake.csv – the dataset file containing fake news articles

The main directory contains:
app.py – the Streamlit web app file
model.pkl – the saved trained machine learning model
confusion_matrix.png – visualization of model evaluation
requirements.txt – list of dependencies to install
README.md – project documentation
data folder

4. Setup Instructions
Step 1: Download all files
app.py
model.pkl
requirements.txt
confusion_matrix.png
And create a data/ folder containing the dataset files (True.csv, Fake.csv)

Step 2: Install dependencies
Open a terminal in the project directory and install the required libraries using:
pip install -r requirements.txt

Step 3: Run the Streamlit app
Once all files are in place and dependencies are installed, run the following command:
streamlit run app.py

This will launch the Fake News Detection web application in your browser.
Enter any news headline or paragraph, and the model will predict whether it is Real or Fake, along with its confidence level.


