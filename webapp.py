import pandas as pd
import re
import numpy as np
import joblib
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st

st.title("Sentiment Analisys")
uploaded_file = st.file_uploader("pilih file preprocessing", key=1)
if uploaded_file is not None:
	maps = pd.read_csv(uploaded_file, encoding = 'unicode_escape')
	st.write("DATA ASLI")
	maps
	#perkecil huruf
	maps['review'] = maps['review'].str.lower()
	st.write("CASE FOLDING")
	maps

	#cleaning data

	def remove_text(text):
    # remove tab, new line, ans back slice
	    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
	    # remove non ASCII (emoticon, chinese word, .etc)
	    text = text.encode('ascii', 'replace').decode('ascii')
	    # remove mention, link, hashtag
	    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
	    # remove incomplete URL
	    return text.replace("http://", " ").replace("https://", " ")
	maps['review'] = maps['review'].apply(remove_text)
	#remove number
	def remove_number(text):
	    return  re.sub(r"\d+", "", text)
	maps['review'] = maps['review'].apply(remove_number)
	def remove_translate(text):
	    return  re.sub(r"translated by google", "", text)
	maps['review'] = maps['review'].apply(remove_translate)
	def remove_terjemahan(text):
	    return  re.sub(r"diterjemahkan oleh google", "", text)
	maps['review'] = maps['review'].apply(remove_terjemahan)
	def remove_original(text):
	    return  re.sub(r"original", "", text)
	maps['review'] = maps['review'].apply(remove_original)

	def remove_punctuation(text):
	    return text.translate(str.maketrans("","",string.punctuation))
	maps['review'] = maps['review'].apply(remove_punctuation)
	def remove_whitespace_LT(text):
	    return text.strip()
	maps['review'] = maps['review'].apply(remove_whitespace_LT)
	def remove_whitespace_multiple(text):
	    return re.sub('\s+',' ',text)
	maps['review'] = maps['review'].apply(remove_whitespace_multiple)
	def remove_singl_char(text):
	    return re.sub(r"\b[a-zA-Z]\b", "", text)
	maps['review'] = maps['review'].apply(remove_singl_char)
	st.write("DATA CLEANING")
	maps

	#tokenisasi
	def tokenization(text):
	  text =re.split("\W", text)
	  return text

	maps['TOKENIZATION'] = maps['review'].apply(lambda x: tokenization(x.lower()))
	st.write("HASIL TOKENISASI")
	maps['TOKENIZATION']

	#stopword
	nltk.download('stopwords')
	stopword = nltk.corpus.stopwords.words("indonesian")

	def remove_stopwords(text):
	  text = [word for word in text if word not in stopword]
	  return text

	maps['STOPWORD'] = maps['TOKENIZATION'].apply(lambda x: remove_stopwords(x))
	st.write("HASIL STOPWORD")
	maps['STOPWORD']

	#pembersihan akhir

	stop_removal = maps[['STOPWORD']]

	def fit_stopwords(text):
	  text = np.array(text)
	  text = ' '.join(text)
	  return text

	maps['STOPWORD'] = maps['STOPWORD'].apply(lambda x: fit_stopwords(x))
	st.write("HASIL AKHIR")
	maps

uploaded_file2 = st.file_uploader("pilih file", key=2)
if uploaded_file2 is not None:
	maps = pd.read_csv(uploaded_file2, encoding = 'unicode_escape')
	st.write("DATA ASLI")
	maps

	st.write("ANALISIS")
	def convert(polarity):
	  if polarity == 'positif':
	    return 1
	  elif polarity == 'netral':
	    return 0
	  else:
	    return -1

	maps['polarity'] = maps['label'].apply(convert)

	X = maps['reviq']
	y = maps['polarity']
	st.write("HASIL POLARITY")
	 


	  




