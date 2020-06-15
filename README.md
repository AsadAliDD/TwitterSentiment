# TwitterSentiment

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/AsadAliDD/TwitterSentiment/graphs/commit-activity) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

TwitterSentiment is a **Keyword based Twitter Sentiment Analyzer**. It uses tweets based on the input provided by the user to generate a Rudimentry Sentiment Report.



## Table of Contents

- [TwitterSentiment](#twittersentiment)
  * [Dataset](#dataset)
  * [Methodology](#Methodology)


## Dataset

The Sentiment Model is trained on **sentiment140** dataset. The dataset contains: 
* 800k Positive Tweets
* 800k Negative Tweets 

https://www.kaggle.com/kazanova/sentiment140


## Methodology

The 1.4 Million tweets are preprocessed using the following steps:
* LowerCase the letters
* Replacing the urls with "URL"
* Removing Usernames (@donaldtrump)
* Removing Special Characters i.e: Non-Alpha Numeric Characters
* Removing Stopwords
* Word lemmatization

The preprocessed tweets are then vectorized using **Tf-idf**. The vectorized tweets are used as a input for the Support Vector Machine Classifer. 


##