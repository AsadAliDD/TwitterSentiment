import GetOldTweets3 as got
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import plotly.graph_objects as go

import streamlit as st
import datetime


count = 100


@st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True)
def queryTweet(tweet):
    text_query = tweet

    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
        text_query).setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    # Creating list of chosen tweet data
    text_tweets = [[tweet.date, tweet.text] for tweet in tweets]

    st.subheader('First tweet result of your query:')
    st.info(text_tweets[0][0])
    st.success(text_tweets[0][1])

    df = pd.DataFrame(text_tweets, columns=['Date', 'Tweets'])


@st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True)
def getTweets(query, count):
    text_query = query
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
        text_query).setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    text_tweets = [[tweet.date, tweet.text] for tweet in tweets]
    df = pd.DataFrame(text_tweets, columns=['Date', 'Tweets'])
    return df


@st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True)
def preprocess(tweet):
    # LowerCase
    tweet = tweet.lower()

    # Replacing URL
    tweet = tweet.replace(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "URL")

    # Removing Username
    tweet = tweet.replace(r'@[^\s]+', "")

    # Removing Non-Alpha Numeric Chars
    tweet = tweet.replace(r'[^A-Za-z0-9 ]+', "")
    stop_words = stopwords.words('english')
    text_tokens = word_tokenize(tweet)
    tokens_without_sw = [
        word for word in text_tokens if not word in stop_words]

    # Lementize
    wordlem = WordNetLemmatizer()
    tokens_without_sw = [wordlem.lemmatize(word) for word in tokens_without_sw]
    filtered_sentence = (" ").join(tokens_without_sw)

    return filtered_sentence


@st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True)
def load_models():

    # Load the vectoriser.
    file = open('Models/tfidf-ngram-(1,3).pickle', 'rb')
    vectorizer = pickle.load(file)
    file.close()

    # Load the LR Model.

    file = open('Models/svc.pickle', 'rb')

    lr = pickle.load(file)
    file.close()

    return vectorizer, lr


@st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True)
def predict(vectorizer, model, tweets):

    print("----------------PreProcessing--------------------------")
    preproc = []
    for tweet in tweets:
        preproc.append(preprocess(tweet))

    print("----------------Vectorising--------------------------")
    vect = vectorizer.transform(preproc)

    print("----------------Predicting--------------------------")
    sent = model.predict(vect)

    data = []
    for text, pred in zip(tweets, sent):
        data.append((text, pred))

    df = pd.DataFrame(data, columns=["Tweets", "Sentiment"])
    df = df.replace([0, 1], ["Negative", "Positive"])

    return df


def main():
    # page title
    st.title('Twitter Sentiment Analsysis')
    activities = ['Analyze Tweets', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)
    if choice == "Analyze Tweets":

        st.subheader('Input a tweet query')

        # user query
        user_input = st.text_input("Enter a tweet", "Type Here.")
        user_input2 = st.text_input(
            "Enter a tweet to query against", "Type Here.")
        if st.button("Submit"):
            text_query = user_input
            queryTweet(text_query)
            tw2 = getTweets(user_input2, 100)
            tw1 = getTweets(user_input, 100)

            vect, model = load_models()
            tw1_pred = predict(vect, model, tw1["Tweets"].tolist())
            tw2_pred = predict(vect, model, tw2["Tweets"].tolist())

            tw1_pred["Date"] = tw1["Date"]
            tw2_pred["Date"] = tw2["Date"]
            # tdf["Date"]=df["Date"]

            st.subheader('First 100 results')
            st.dataframe(tw1_pred)

            # plot
            st.subheader("Scatter Plot")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tw1_pred["Date"], y=tw1_pred["Sentiment"], name='First Tweet'))
            st.plotly_chart(fig)

            st.subheader("Pie Chart")
            val = tw1_pred["Sentiment"].value_counts().values
            val2 = tw2_pred["Sentiment"].value_counts().values

            # pie chart
            fig = go.Figure()
            fig.add_trace(go.Pie(labels=['Negative', 'Positive'],
                                 values=val, name='First Tweet'))
            st.plotly_chart(fig)

           # bar chart
            st.subheader("Bar Chart")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=['Negative', 'Positive'], y=val, name='First Tweet'))
            fig.add_trace(
                go.Bar(x=['Negative', 'Positive'], y=val2, name='Second Tweet'))

            fig.update_layout(title="{} v {}".format("Tweet 1", "Tweet 2"), title_x=0.5,
                              xaxis_title='Sentiment',
                              yaxis_title='Number of Tweets')
            st.plotly_chart(fig)

    elif choice == "About":
        st.subheader("Orientation Project for Team Rigel")
        st.markdown(
            "Built by [Paul](https://github.com/talentmavingire/)" " ," " [Asad](https://github.com/AsadAliDD/)"" ,and" " [Maaz](https://github.com/maazzzzz/)")


if __name__ == '__main__':
    main()
