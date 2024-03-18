import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import pickle

# Load the data
@st.cache
def load_data():
    dataset = pd.read_csv('dataset_undersampled.csv')
    return dataset

# Sidebar
st.sidebar.title('Options')
selected_option = st.sidebar.selectbox('Choose an Option', ['Dataset Overview', 'Data Preprocessing', 'BERT Model'])

# Main content
st.title('Mental Disorders Identification')

if selected_option == 'Dataset Overview':
    # Load the data
    dataset = load_data()
    st.write('Dataset Overview')
    st.write(dataset.head())
    st.write(f'Shape of the dataset: {dataset.shape}')

    # Plot distribution of subreddit categories
    st.write('Distribution of Subreddit Categories')
    subreddit_counts = dataset['subreddit'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=subreddit_counts.index, y=subreddit_counts.values)
    plt.xticks(rotation=45)
    st.pyplot()

elif selected_option == 'Data Preprocessing':
    st.write('Data Preprocessing')

    # Load the data
    dataset = load_data()

    # Text Preprocessing
    def preprocess_text(text):
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.strip()  # Remove leading/trailing whitespaces
        return text

    dataset['title'] = dataset['title'].apply(preprocess_text)

    # Show preprocessed text
    st.write('Preprocessed Text:')
    st.write(dataset['title'].head())

elif selected_option == 'BERT Model':
    st.write('BERT Model')

    # Load BERT tokenizer
    with open('bert_tokenizer1.pkl', 'rb') as f:
        bert_tokenizer = pickle.load(f)

    # Load BERT model
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    bert_model.load_weights('bert_model_weights1.h5')

    # Input text
    input_text = st.text_area('Enter Text:', '')

    # Perform prediction
    if st.button('Predict'):
        input_text_tokenized = bert_tokenizer.encode(input_text,
                                                     truncation=True,
                                                     padding='max_length',
                                                     return_tensors='tf')
        bert_predict = bert_model(input_text_tokenized)
        bert_output = tf.nn.softmax(bert_predict[0], axis=-1)
        mental_illness_label = ['BPD', 'Anxiety', 'Depression', 'Mental Illness']
        label = tf.argmax(bert_output, axis=1)
        label = label.numpy()[0]
        st.write(f'Predicted Mental Disorder: {mental_illness_label[label]}')

# Footer
st.sidebar.title('About')
st.sidebar.info('This app is created by [Your Name]')
