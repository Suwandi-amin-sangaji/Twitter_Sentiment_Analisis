import streamlit as st
import textblob
import re
from tweepy import OAuthHandler
import json
import csv
import matplotlib
import base64
import tweepy
import numpy as np
import warnings
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import joblib
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Halaman "Home"


def home():
    # Kode untuk menampilkan konten halaman "Home"
    with st.sidebar:
        st.image(Image.open('Logo1.jpg'))
        st.caption('© SoorngDev 2023')

    # Tambahkan konten halaman Home sesuai kebutuhan
    html_temp = """
    <div style="background-color:#53c2ea; text-align:center;"><p style="color:black;font-size:40px;padding:9px">Selamat Datang Di</p></div>
    <div style="background-color:#53c2ea; text-align:center;"><p style="color:black;font-size:35px;padding:9px"> Twitter Sentiment Analysis LOGISTIC REGRESSION</p></div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

# Halaman "Scraping"


def scraping():
    with st.sidebar:
        st.image(Image.open('Logo1.jpg'))
        st.caption('© SoorngDev 2023')
    st.title("Scraping Data")
    ################# Twitter API Connection #######################
    consumer_key = "P1DzgreWqH4jJRRZGYjVsnyOl"
    consumer_secret = "sZlrqDzxumdhvvn3JM4r5G8VSyIeU8Nq2jf1bL4yDWwaKFxxD4"
    access_token = "1217082892586602496-if2NPUJqGvhYkUyRBtfefVRZ4czPca"
    access_token_secret = "gz5adcJrQW6gpl6KfErOwAONr1o71Dp3WCd5mkJT7r0o0"

    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    ################################################################

    df = pd.DataFrame(
        columns=["Date", "User", "Tweet",])

    # Write a Function to extract tweets:
    def get_tweets(Topic, Count):
        i = 0
        # my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search_tweets, q=Topic, count=100, lang="id", exclude='retweets').items():
            # time.sleep(0.1)
            # my_bar.progress(i)
            df.loc[i, "Date"] = tweet.created_at
            df.loc[i, "User"] = tweet.user.name
            df.loc[i, "Tweet"] = tweet.text
            # df.to_csv("TweetDataset.csv",index=False)
            # df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i = i+1
            if i > Count:
                break
            else:
                pass

    topik_list = ["paylater", "paylater shopee", "paylater lazada"]

    # Dropdown untuk memilih topik
    topic = st.selectbox("Pilih topik:", topik_list, index=0)

    # Input jumlah tweet yang ingin diambil
    num_tweets = st.number_input("Jumlah tweet yang ingin diambil:",
                                 min_value=100, max_value=1000, value=100, step=100)

    # Button untuk mengambil tweet
    if st.button("Ambil Tweet"):
        # Menampilkan hasil seleksi Topic
        st.write("Anda telah memilih Topik:", topic)

        if topic:
            # Panggil fungsi untuk mengambil data. Berikan Topic dan nama file untuk menyimpan data
            with st.spinner("Mohon tunggu, Tweets sedang diekstrak"):
                # Merubah parameter Topic menjadi topic
                get_tweets(topic, Count=num_tweets)
            st.success('Scraping Data Sukses !!!!')
            st.write(df.head(1000))

        # Menambahkan tombol download CSV
        csv_data = df.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{topic}_tweets.csv">Unduh Data CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


# Halaman "Labeling & Pre-Processing"


def labeling_preprocessing():
    with st.sidebar:
        st.image(Image.open('Logo1.jpg'))
        st.caption('© SoorngDev 2023')
    st.title("Labeling & Pre-Processing")
    # Kode untuk menampilkan konten halaman "Labeling & Pre-Processing"
    # Menghilangkan stopwords dalam bahasa Inggris

    stop_words = set(stopwords.words('english'))

    # Membuat objek PorterStemmer
    ps = PorterStemmer()

    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def clean_tweet(tweet):
        # Menghapus karakter yang tidak diinginkan
        tweet = re.sub(
            '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower())
        return tweet

    def case_folding(tweet):
        # Mengubah teks menjadi lowercase
        return tweet.lower()

    def tokenizing(tweet):
        # Melakukan tokenisasi pada teks
        return word_tokenize(tweet)

    def normalization(tokens):
        # Melakukan normalisasi kata-kata
        normalized_tokens = []
        for token in tokens:
            normalized_token = ps.stem(token)
            normalized_tokens.append(normalized_token)
        return normalized_tokens

    def removal_stopwords(tokens):
        # Menghapus stopwords dari teks
        tokens_without_stopwords = [
            token for token in tokens if token not in stop_words]
        return tokens_without_stopwords

    def stemming(tokens):
        # Melakukan stemming pada kata-kata
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = ps.stem(token)
            stemmed_tokens.append(stemmed_token)
        return stemmed_tokens

    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk preprocessing", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)
        st.write("Data Awal:")
        st.write(df.head(1000))
        # Panggil fungsi preprocessing saat tombol ditekan

    def preprocessing():
        st.write("Start Pre-processing")
        st.write("| cleaning...")
        time.sleep(1)  # Simulasi waktu pemrosesan
        st.write("| case folding...")
        time.sleep(1)  # Simulasi waktu pemrosesan
        st.write("| tokenizing...")
        time.sleep(1)  # Simulasi waktu pemrosesan
        st.write("| normalization...")
        time.sleep(1)  # Simulasi waktu pemrosesan
        st.write("| removal stopwords...")
        time.sleep(1)  # Simulasi waktu pemrosesan
        st.write("| stemming...")
        time.sleep(1)  # Simulasi waktu pemrosesan
        st.write("Finish Pre-processing")

    if st.button("Mulai Pre-processing"):
        preprocessing()
        df['clean_tweet'] = df['Tweet'].apply(clean_tweet)
        df['clean_tweet'] = df['clean_tweet'].apply(case_folding)
        df['label'] = df['clean_tweet'].apply(analyze_sentiment)
        df['tokens'] = df['clean_tweet'].apply(tokenizing)
        df['normalized_tokens'] = df['tokens'].apply(normalization)
        df['tokens_without_stopwords'] = df['normalized_tokens'].apply(
            removal_stopwords)
        df['stemmed_tokens'] = df['normalized_tokens'].apply(stemming)
        # Mengurutkan label ke paling akhir
        cols = df.columns.tolist()
        cols.remove('label')
        cols.append('label')
        df = df[cols]
        st.write("Hasil Preprocessing:")
        st.write(df.head(1000))
        temp_file = df.to_csv(index=False)
        b64 = base64.b64encode(temp_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="hasil_preprocessing.csv">Download Hasil Preprocessing</a>'
        st.markdown(href, unsafe_allow_html=True)
    # else:
    #     st.warning("Silakan pilih file CSV untuk melakukan preprocessing.")

# Halaman "Modeling"


def modeling():
    with st.sidebar:
        st.image(Image.open('Logo1.jpg'))
        st.caption('© SoorngDev 2023')
    st.title("Modeling & Visualisasi")
    # Kode untuk menampilkan konten halaman "Modeling"
    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk preprocessing", type=["csv"])

    # Jika file CSV dipilih
    if uploaded_file is not None:

        # Membaca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file)

        # Menampilkan informasi tentang file yang diunggah
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)

        # # Menampilkan data awal dari DataFrame
        st.write("Data Awal:")
        st.write(df.head(1000))

        # Fungsi untuk melakukan analisis sentimen menggunakan Logistic Regression

        # Menambahkan tombol untuk memulai preprocessing dan training model
        if st.button("Mulai Preprocessing dan Training Model"):

            # Memisahkan fitur dan label
            X = df['clean_tweet']
            y = df['label']

            # Melakukan vectorization pada teks
            vectorizer = CountVectorizer()
            X_vectorized = vectorizer.fit_transform(X)

            # Membagi data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42)

            # Melatih model Logistic Regression
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Memprediksi label pada data uji
            y_pred = model.predict(X_test)

            # Menghitung akurasi, F1 score, dan confusion matrix model
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            # Menampilkan akurasi, F1 score, dan confusion matrix
            # st.write("Akurasi Model:")
            # st.write(accuracy)
            # st.write("F1 Score:")
            # st.write(f1)
            st.write("Accuracy: {:.2f}%".format(accuracy * 100))
            st.write("Precision: {:.2f}%".format(precision * 100))
            st.write("Recall: {:.2f}%".format(recall * 100))
            st.write("F1-score: {:.2f}%".format(f1 * 100))

            # Visualisasi akurasi dan F1 score
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].bar(['Akurasi'], [accuracy])
            ax[0].bar(['F1 Score'], [f1])
            ax[0].set_ylabel('Nilai')
            ax[0].set_title('Akurasi dan F1 Score Model')
            ax[0].set_ylim([0, 1])
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)

            # Visualisasi confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1])
            ax[1].set_xlabel('Prediksi')
            ax[1].set_ylabel('Aktual')
            ax[1].set_title('Confusion Matrix')
            plt.tight_layout()

            # Menampilkan visualisasi data
            st.pyplot(fig)

            # Menggabungkan semua teks menjadi satu string
            all_text = ' '.join(df['clean_tweet'].values)

            # Membuat objek WordCloud
            wordcloud = WordCloud(width=800, height=400, max_words=150,
                                  background_color='white').generate(all_text)

            # Visualisasi WordCloud
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title('WordCloud - Kata yang Sering Muncul')
            ax.axis('off')

            # Menampilkan visualisasi data
            st.pyplot(fig)

            # Menghitung kata yang sering muncul
            from collections import Counter

            # Mengubah string menjadi list kata-kata
            words_list = all_text.split()

            # Menghitung jumlah kemunculan setiap kata
            word_count = Counter(words_list)

            # Mengambil 10 kata yang paling sering muncul
            common_words = word_count.most_common(10)

            # Mengambil kata dan jumlah kemunculannya
            words = [word for word, count in common_words]
            count = [count for word, count in common_words]

            # Menggabungkan semua teks tweet positif dan negatif menjadi satu string
            all_positive_tweets = ' '.join(
                df[df['label'] == 'Positive']['clean_tweet'])
            all_negative_tweets = ' '.join(
                df[df['label'] == 'Negative']['clean_tweet'])

            a = len(df[df["label"] == "Positive"])
            b = len(df[df["label"] == "Negative"])
            c = len(df[df["label"] == "Neutral"])
            # d = len(df[df["label"] == "Mixed"])

            # Membuat diagram batang
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(["Positive", "Negative", "Neutral"], [a, b, c])
            ax.set_title('Jumlah Data untuk Setiap Sentimen')
            ax.set_xlabel('Sentimen')
            ax.set_ylabel('Jumlah Data')
            # Menampilkan visualisasi data
            st.pyplot(fig)

            # Membuat diagram batang
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(words, count)
            ax.set_title('Kata yang Sering Muncul')
            ax.set_xlabel('Kata')
            ax.set_ylabel('Jumlah Kemunculan')
            # Menampilkan visualisasi data
            st.pyplot(fig)

        # Piechart
            a = len(df[df["label"] == "Positive"])
            b = len(df[df["label"] == "Negative"])
            c = len(df[df["label"] == "Neutral"])
            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [a, b, c]
            colors = ['#66b3ff', '#ff9999', '#99ff99']
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, colors=colors, labels=labels,
                    autopct='%1.1f%%', startangle=90)
            # Draw circle
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.axis('equal')
            plt.title("Persentase Sentimen")
            plt.tight_layout()
            # Menampilkan visualisasi data
            st.pyplot(fig1)

            # Membuat WordCloud untuk tweet positif

            wordcloud_positive = WordCloud(
                width=800, height=400, background_color='white').generate(all_positive_tweets)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.axis('off')
            plt.title('WordCloud untuk Tweet Positif')
            # Menampilkan visualisasi data
            st.pyplot(plt)

            # Mengecek apakah ada sentimen negatif
            if 'Negative' in df['label'].values:

                wordcloud_negative = WordCloud(
                    width=800, height=400, background_color='white').generate(all_negative_tweets)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud_negative, interpolation='bilinear')
                plt.axis('off')
                plt.title('WordCloud untuk Tweet Negatif')
                # Menampilkan visualisasi data
                st.pyplot(plt)

# Halaman "Prediksi"


def prediksi():
    with st.sidebar:
        st.image(Image.open('Logo1.jpg'))
        st.caption('© SoorngDev 2023')
    st.title("Prediksi")

    # Kode untuk menampilkan konten halaman "Prediksi"
    # Upload file CSV
    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk preprocessing", type=["csv"])
    if uploaded_file is not None:
        # Baca file CSV
        df = pd.read_csv(uploaded_file)

        # Preprocessing data
        # Contoh: Mengubah teks menjadi lowercase dan menghapus tanda baca
        df['clean_tweet'] = df['clean_tweet'].str.lower()
        df['clean_tweet'] = df['clean_tweet'].str.replace('[^\w\s]', '')

        # Pembagian data latih dan data uji
        X = df['clean_tweet']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Vectorizer
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Melatih model Logistic Regression
        model = LogisticRegression()
        model.fit(X_train_vec, y_train)

        # Evaluasi model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Menampilkan hasil evaluasi model
        st.write("Accuracy: {:.2f}%".format(accuracy * 100))
        st.write("Precision: {:.2f}%".format(precision * 100))
        st.write("Recall: {:.2f}%".format(recall * 100))
        st.write("F1-score: {:.2f}%".format(f1 * 100))

        # Fungsi untuk preprocessing teks
        def preprocess_text(text):
            text = text.lower()
            text = text.replace('[^\w\s]', '')
            return text

        # Inputan teks baru untuk prediksi perkata
        input_text = st.text_area(
            "Masukkan teks untuk diprediksi sentimennya:", height=100)

        if st.button("Prediksi"):
            # Preprocessing teks inputan
            input_text = preprocess_text(input_text)

            # Melakukan prediksi perkata
            input_text_vec = vectorizer.transform([input_text])
            sentiment_probs = model.predict_proba(input_text_vec)[0]
            sentiment_confidence = max(sentiment_probs) * 100

            st.write("Sentimen prediksi untuk teks baru: ",
                     model.predict(input_text_vec)[0])
            st.write("Confidence: {:.2f}%".format(sentiment_confidence))


# List halaman
pages = ["Home", "Scraping Data",
         "Labeling & Pre-Processing", "Modeling & Visualisasi", "Prediksi"]

# Menampilkan menu navigasi di sidebar
page = st.sidebar.selectbox("Menu Sentiment Analisis", pages)

# Menampilkan konten halaman berdasarkan pilihan pengguna
if page == "Home":
    home()
elif page == "Scraping Data":
    scraping()
elif page == "Labeling & Pre-Processing":
    labeling_preprocessing()
elif page == "Modeling & Visualisasi":
    modeling()
elif page == "Prediksi":
    prediksi()
