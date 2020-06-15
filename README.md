# TwitterSentiment

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/AsadAliDD/TwitterSentiment/graphs/commit-activity) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![GitHub followers](https://img.shields.io/github/followers/asadalidd?label=Followers&style=flat)
![GitHub contributors](https://img.shields.io/github/contributors/asadalidd/twittersentiment)

TwitterSentiment is a **Keyword based Twitter Sentiment Analyzer**. It uses tweets based on the input provided by the user to generate a Rudimentry Sentiment Report.

Orientation Project for **Hack Your Own**


https://twittersa-hyo.oa.r.appspot.com/

![UI](Images/ui.gif)



## Table of Contents

- [TwitterSentiment](#twittersentiment)
  * [Dataset](#dataset)
  * [Methodology](#Methodology)
  * [Docker Container](#Docker)
  * [Usage](#Usage)
  * [Dependencies](#Dependencies)


## Dataset

The Sentiment Model is trained on **sentiment140** dataset. 
https://www.kaggle.com/kazanova/sentiment140

The dataset contains: 
* 800k Positive Tweets
* 800k Negative Tweets 


## Methodology

The 1.4 Million tweets are preprocessed using the following steps:
* LowerCase the letters
* Replacing the urls with "URL"
* Removing Usernames (@donaldtrump)
* Removing Special Characters i.e: Non-Alpha Numeric Characters
* Removing Stopwords
* Word lemmatization

The preprocessed tweets are then vectorized using **Tf-idf**. The vectorized tweets are used as a input for the Support Vector Machine Classifer. 


## Docker

> :warning:  Docker Linux is needed for this!

`docker pull realdexter/twitter_sentiment:v1`

`docker run  -p local_port:8501 realdexter/twitter_sentiment:v1`

*local_port* is the Port you want to map to the exposed port of the container.

Visit https://localhost:8501



## Usage (Locally)

> :warning:  Model files not included in git repo.


`git clone https://github.com/AsadAliDD/TwitterSentiment`

`pip3 install -r requirements.txt`

`streamlit run SentimentAnalysis.py`



## Dependencies 

* Python3
* Streamlit
* Nltk
* Sklearn
* Pandas
* Numpy
* GetOldTweets3
* Plotly






