#!/usr/bin/env python
# coding: utf-8

# In[68]:


import GetOldTweets3 as got
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import plotly.graph_objects as go


# In[2]:


get_ipython().run_cell_magic('time', '', "\ntext_query = 'USA Election 2020'\ncount = 100\n# Creation of query object\ntweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setMaxTweets(count)\n# Creation of list that contains all tweets\ntweets = got.manager.TweetManager.getTweets(tweetCriteria)")


# In[3]:


# Creating list of chosen tweet data
text_tweets = [[tweet.date, tweet.text] for tweet in tweets]


# In[4]:


text_tweets[0]


# In[5]:


df = pd.DataFrame(text_tweets, columns =['Date', 'Tweets']) 


# In[6]:


def getTweets(query,count):
    text_query = query
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    text_tweets = [[tweet.date, tweet.text] for tweet in tweets]
    df = pd.DataFrame(text_tweets, columns =['Date', 'Tweets']) 
    return df


# In[10]:


def preprocess(tweet):
    #LowerCase 
    tweet=tweet.lower()

    #Replacing URL
    tweet=tweet.replace(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "URL")

    #Removing Username
    tweet=tweet.replace(r'@[^\s]+',"")

    #Removing Non-Alpha Numeric Chars
    tweet=tweet.replace(r'[^A-Za-z0-9 ]+',"")


    stop_words = stopwords.words('english') 
    text_tokens = word_tokenize(tweet)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    

    #Lementize
    wordlem = WordNetLemmatizer()
    tokens_without_sw=[wordlem.lemmatize(word) for word in tokens_without_sw]
    filtered_sentence = (" ").join(tokens_without_sw)



    return filtered_sentence


def load_models():
    
    # Load the vectoriser.
    file = open('../Models/tfidf-ngram-(1,3).pickle', 'rb')
    # file = open('C:/Users/mavin/Desktop/TwitterSentiment/Models/tfidf-ngram-(1,3).pickle', 'rb')
    vectorizer = pickle.load(file)
    file.close()
    
    # Load the LR Model.
    
    file = open('../Models/svc.pickle', 'rb')
    # file = open('C:/Users/mavin/Desktop/TwitterSentiment/Models/svc.pickle', 'rb')
    
    lr = pickle.load(file)
    file.close()
    
    return vectorizer, lr



def predict(vectorizer,model,tweets):

    print ("----------------PreProcessing--------------------------")
    preproc=[]
    for tweet in tweets:
        preproc.append(preprocess(tweet))

    print ("----------------Vectorising--------------------------")
    vect=vectorizer.transform(preproc)
    
    print ("----------------Predicting--------------------------")
    sent=model.predict(vect)


    data = []
    for text, pred in zip(tweets, sent):
        data.append((text,pred))

    df=pd.DataFrame(data,columns=["Tweets","Sentiment"])
    df = df.replace([0,1], ["Negative","Positive"])

    return df


# In[8]:


get_ipython().run_cell_magic('time', '', "tw1=getTweets('USA Election 2020',100)\ntw2=getTweets('USA Election 2016',100)")


# In[14]:


vect,model=load_models()
tw1_pred=predict(vect,model,tw1["Tweets"].tolist())
tw2_pred=predict(vect,model,tw2["Tweets"].tolist())


# In[15]:


tw1_pred["Date"]=tw1["Date"]
tw2_pred["Date"]=tw2["Date"]
# tdf["Date"]=df["Date"]


# In[16]:


tw1_pred


# In[96]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=tw1_pred["Date"],y=tw1_pred["Sentiment"],name='USA Election 2020'))
fig.show()


# In[98]:


val=tw1_pred["Sentiment"].value_counts().values
val2=tw2_pred["Sentiment"].value_counts().values


# In[99]:


fig=go.Figure()
fig.add_trace(go.Pie(labels=['Negative','Positive'],values=val,name='Election 2020'))
fig.show()


# In[100]:


fig=go.Figure()
fig.add_trace(go.Bar(x=['Negative','Positive'],y=val,name='Election 2020'))
fig.add_trace(go.Bar(x=['Negative','Positive'],y=val2,name='Election 2016'))

fig.update_layout(title='Election 2016 v Election 2020',title_x=0.5,
                   xaxis_title='Sentiment',
                   yaxis_title='Number of Tweets')

fig.show()


# In[95]:


val=tw1_pred["Sentiment"].value_counts().values
val2=tw2_pred["Sentiment"].value_counts().values


# In[101]:





# In[ ]:




