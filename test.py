import numpy as np 
import pandas as pd 
import re
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from time import sleep
import matplotlib.pyplot as plt
import streamlit as st

st.title("Test Sentiment!")

uploaded_file = st.file_uploader("pilih file")
if uploaded_file is not None:
	maps = pd.read_csv(uploaded_file)
	st.write("DATA ASLI")
	maps
  
	maps['review'] = maps['review'].str.lower()
	st.write("HURUF KECIL")
	maps

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
	st.write("MENGHILANGKAN TANDA BACA")
	maps

	nltk.download('punkt')

	# NLTK word rokenize 
	def word_tokenize_wrapper(text):
	    return word_tokenize(text)
	maps['tokenizing'] = maps['review'].apply(word_tokenize_wrapper)
	st.write("HASIL TOKENIZING")
	maps['tokenizing']

	def freqDist_wrapper(text):
		return FreqDist(text)

	maps['token_freq'] = maps['tokenizing'].apply(freqDist_wrapper)

	print('Frequency Tokens : \n') 
	print(maps['token_freq'].head().apply(lambda x : x.most_common()))

	nltk.download('stopwords')
	list_stopwords = stopwords.words('indonesian')

	list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

	txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)


	list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
	list_stopwords = set(list_stopwords)

	def stopwords_removal(words):
	    return [word for word in words if word not in list_stopwords]

	maps['stopword_remove'] = maps['tokenizing'].apply(stopwords_removal) 


	print(maps['stopword_remove'].head())

	normalizad_word = pd.read_csv("normalisasi.csv")

	normalizad_word_dict = {}

	for index, row in normalizad_word.iterrows():
	    if row[0] not in normalizad_word_dict:
	        normalizad_word_dict[row[0]] = row[1] 

	def normalized_term(document):
	    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

	maps['normalisasi'] = maps['stopword_remove'].apply(normalized_term)
	st.write("HASIL NORMALISASI")

	maps['normalisasi']

	# create stemmer
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()

	# stemmed
	def stemmed_wrapper(term):
	    return stemmer.stem(term)

	term_dict = {}

	for document in maps['normalisasi']:
	    for term in document:
	        if term not in term_dict:
	            term_dict[term] = ' '
	            
	print(len(term_dict))
	print("------------------------")

	for term in term_dict:
	    term_dict[term] = stemmed_wrapper(term)
	    print(term,":" ,term_dict[term])
	    
	print(term_dict)
	print("------------------------")


	# apply stemmed term to dataframe
	def get_stemmed_term(document):
	    return [term_dict[term] for term in document]

	maps['stem'] = maps['normalisasi'].swifter.apply(get_stemmed_term)
	print(maps['stem'])
	st.write("HASIL STEMMMING")
	maps['stem']

	  
		
