import GetOldTweets3 as got
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
import datetime

import time


count = 100




# @st.cache(suppress_st_warning=True)
# @st.cache(allow_output_mutation=True)
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


# @st.cache(suppress_st_warning=True)
# @st.cache(allow_output_mutation=True)
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


# @st.cache(suppress_st_warning=True)
# @st.cache(allow_output_mutation=True)
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


# @st.cache()
def load_models():

    start=time.time()
    # Load the vectoriser.
    file = open('./Models/tfidf-ngram-(1,3).pickle', 'rb')
    vectorizer = pickle.load(file)
    file.close()

    # Load the LR Model.

    file = open('./Models/svc.pickle', 'rb')

    lr = pickle.load(file)
    file.close()

    end = time.time()
    print("Loading Model Took: ",end - start)
    return vectorizer, lr


# @st.cache(allow_output_mutation=True)
def predict(vectorizer, model, tweets):

    start=time.time()
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

    end = time.time()
    print("Predicting Took: ",end - start)
    
    return df


def main():
    # page title
    st.title('Twitter Sentiment Analsysis')
    activities = ['Analyze Tweets', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)


    #Loading Models
    if choice == "Analyze Tweets":

        flag=st.sidebar.checkbox('Add Keyword')
        st.subheader('Input a tweet query')

        # user query
        user_input = st.text_input("Keyword", "Type Here.")
        if flag:
            user_input2 = st.text_input(
                "Another Keyword", "Type Here.")


        if st.button("Submit"):

            start=time.time()
    

            text_query = user_input
            queryTweet(text_query)
            tw1 = getTweets(user_input, 100)

            vect, model = load_models()
            
            tw1_pred = predict(vect, model, tw1["Tweets"].tolist())
            tw1_pred["Date"] = tw1["Date"]

            st.subheader(user_input)
            st.dataframe(tw1_pred)


            if(flag):
                tw2 = getTweets(user_input2, 100)
                tw2_pred = predict(vect, model, tw2["Tweets"].tolist())
                tw2_pred["Date"] = tw2["Date"]


                st.subheader(user_input2)
                st.dataframe(tw2_pred)

            # tdf["Date"]=df["Date"]

            
            if(flag):
                # scatter plot
                st.subheader("Scatter Plot")
                fig = make_subplots(rows=1, cols=2)   
                fig.add_trace(
                    go.Scatter(
                    x=tw1_pred["Date"], y=tw1_pred["Sentiment"], name=user_input),row=1,col=1)
                fig.add_trace(
                    go.Scatter(
                    x=tw2_pred["Date"], y=tw2_pred["Sentiment"], name=user_input2),row=1,col=2)
                st.plotly_chart(fig)


                # pie chart
                st.subheader(user_input)
                val = tw1_pred["Sentiment"].value_counts().values
                fig = go.Figure()
                fig.add_trace(go.Pie(labels=['Negative', 'Positive'],
                                    values=val, name=user_input))
                st.plotly_chart(fig)

            
                st.subheader(user_input2)
                val2 = tw2_pred["Sentiment"].value_counts().values
                fig = go.Figure()
                fig.add_trace(go.Pie(labels=['Negative', 'Positive'],
                                    values=val2, name=user_input2))
                st.plotly_chart(fig)



                # bar chart
                st.subheader("Bar Chart")
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(x=['Negative', 'Positive'], y=val, name=user_input))
                fig.add_trace(
                    go.Bar(x=['Negative', 'Positive'], y=val2, name=user_input2))

                fig.update_layout(title="{} v {}".format(user_input, user_input2), title_x=0.5,
                                xaxis_title='Sentiment',
                                yaxis_title='Number of Tweets')
                st.plotly_chart(fig)

            else:
                # plot
                st.subheader("Scatter Plot")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tw1_pred["Date"], y=tw1_pred["Sentiment"], name=user_input))
                
                st.plotly_chart(fig)


                # pie chart
                st.subheader("Pie Chart")
                val = tw1_pred["Sentiment"].value_counts().values
                fig = go.Figure()
                fig.add_trace(go.Pie(labels=['Negative', 'Positive'],
                                    values=val, name='First Tweet'))
                st.plotly_chart(fig)

                # bar chart
                st.subheader("Bar Chart")
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(x=['Negative', 'Positive'], y=val, name=user_input))
                # fig.add_trace(
                #     go.Bar(x=['Negative', 'Positive'], y=val2, name=user_input2))

                fig.update_layout(title=user_input, title_x=0.5,
                                xaxis_title='Sentiment',
                                yaxis_title='Number of Tweets')
                st.plotly_chart(fig)


            end = time.time()
            print("Total Time: ",end - start)
    

    elif choice == "About":
        st.subheader("Orientation Project for Team Rigel")
        st.markdown(
            "Built by [Paul](https://github.com/talentmavingire/)" " ," " [Asad](https://github.com/AsadAliDD/)"" ,and" " [Maaz](https://github.com/maazzzzz/)")


if __name__ == '__main__':
    main()
