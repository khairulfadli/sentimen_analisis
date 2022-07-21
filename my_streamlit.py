import pandas as pd
import re
import numpy as np
import joblib
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
from imblearn.over_sampling import SMOTE
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from time import sleep
import csv
import math
import random
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import itertools
import collections
from collections import Counter
import networkx as nx
import warnings
from nltk import bigrams
from pandas import DataFrame
import pandas as pd
import plotly.graph_objects as go

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Prepocessing','Analisis')
)

if option == 'Home' or option == '':
    st.write("""# Sentiment Analisis""") #menampilkan halaman utama
    st.write("PENGGUNA GOOGLE MAPS DI INDONESIA PADA REVIEW WISATA MENGGUNAKAN NAÏVE BAYES")
elif option == 'Prepocessing':
    st.title("Preprocessing Data")
    uploaded_file = st.file_uploader("pilih file preprocessing", key=1)
    if uploaded_file is not None:
        maps = pd.read_csv(uploaded_file, encoding = 'unicode_escape')
        st.write("DATA ASLI")
        maps
        #perkecil huruf
        maps['review'] = maps['review'].str.lower()
        st.write("CASE FOLDING")
        maps #menampilkan judul halaman dataframe
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
        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # stemmed
        def stemmed_wrapper(term):
            return stemmer.stem(term)

        term_dict = {}

        for document in maps['STOPWORD']:
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

        maps['REVIEW'] = maps['STOPWORD'].swifter.apply(get_stemmed_term)
        print(maps['REVIEW'])

        #pembersihan akhir

        stop_removal = maps[['REVIEW']]

        def fit_stopwords(text):
          text = np.array(text)
          text = ' '.join(text)
          return text

        maps['REVIEW'] = maps['REVIEW'].apply(lambda x: fit_stopwords(x))
        st.write("HASIL AKHIR DATA BERSIH")
        maps
        def convert(polarity):
          if polarity == 'positif':
            return 1
          else:
            return -1
        maps['polarity'] = maps['label'].apply(convert)
        st.write("HASIL POLARITY")
        maps

        def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
             return df.to_csv().encode('utf-8')

        csv = convert_df(maps)

        st.download_button(
             label="Download CSV",
             data=csv,
             file_name='hasil_preprocessing.csv',
             mime='text/csv',
         )

   
elif option == 'Analisis':
    st.write("""## TAHAP ANALISIS DATA""") #menampilkan judul halaman 
    uploaded_file = st.file_uploader("pilih file analisis", key=2)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding = 'unicode_escape')
        st.write("DATA ANALISIS")
        df[['BERSIH','label','polarity']]
        df['word_count'] = df['BERSIH'].apply(lambda x: len(str(x).split(" ")))

        st.write('HITUNG JUMLAH KATA')
        df[['BERSIH','word_count']]

        st.write('HITUNG JUMLAH KARAKTER')
        df['char_count'] = df['BERSIH'].str.len() ## this also includes spaces
        df[['BERSIH','char_count']]

        st.write('HITUNG RATA-RATA PANJANG KATA')
        #to find the average word length
        def avg_word(sentence):
          words = sentence.split()
          return (sum(len(word) for word in words)/len(words))
        df['BERSIH'] = df['BERSIH'].astype(str)
        df['avg_word'] = df['BERSIH'].apply(lambda x: avg_word(x))
        df[['BERSIH','avg_word']]



        st.write('JUMLAH KATA YANG MUNCUL')
        freq = pd.Series(' '.join(df['BERSIH']).split()).value_counts()[:15]
        freq

        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer()
        tfidf.fit_transform(df['BERSIH'].dropna().values.astype('U')) ## Even astype(str) would work
        from sklearn.model_selection import train_test_split
        X = df.BERSIH
        y = df.polarity
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5,random_state = 0)
        #st.write("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),(len(X_train[y_train == -1]) / (len(X_train)*1.)
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score
        def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
            polarity_fit = pipeline.fit(X_train, y_train)
            y_pred = polarity_fit.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("accuracy score: {0:.2f}%".format(accuracy*100))
            return accuracy

        import numpy as np
        cv = CountVectorizer()
        rf = RandomForestClassifier(class_weight="balanced")
        n_features = np.arange(1000,2001,1000)
        def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
            result = []
            st.write(classifier)
            st.write("\n")
            for n in n_features:
                vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
                checker_pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)
                ])
                st.write("Test result for {} features".format(n))
                nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
                result.append((n,nfeature_accuracy))
            return result
        tfidf = TfidfVectorizer()
        st.write("Result for trigram with stop words (Tfidf)\n")
        feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3),stop_words=None)

        from sklearn.metrics import classification_report, accuracy_score
        nb = MultinomialNB()
        tfidf = TfidfVectorizer(max_features=10000,ngram_range=(1, 3))
        pipeline = Pipeline([
                ('vectorizer', tfidf),
                ('classifier', rf)
            ])
        nb.fit = pipeline.fit(X_train, y_train)
        y_pred = nb.fit.predict(X_test)
        df = st.write(classification_report(y_test, y_pred, target_names=['negative','positive']))
        st.write("accuracy score")
        st.write(accuracy_score(y_test, y_pred))
        import scikitplot as skplt
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
        st.pyplot()

       
        


       
                  
