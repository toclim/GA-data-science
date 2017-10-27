# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:46:47 2017

@author: Zhilin
"""

import pandas as pd


url = 'https://raw.githubusercontent.com/toclim/GA-data-science/master/yelp.csv'
yelp = pd.read_csv(url)
yelp.head()
yelp.shape
yelp.stars.value_counts().sort_index()
yelp.isnull().sum()


from textblob import TextBlob
review = TextBlob(yelp.loc[0, 'text'])
review.words


def sentiment_score(text):
    
    blob = TextBlob(text)
    
    return blob.sentiment.polarity


yelp['sentiment'] = yelp.text.apply(sentiment_score)
yelp.head()


def make_features(df):
    df['sentiment_score'] = df.text.apply(sentiment_score)
    return df


yelp = make_features(pd.read_csv(url))
yelp_new = yelp[(yelp.stars==5) | (yelp.stars==3) | (yelp.stars==1)]
yelp_new.head()
yelp_new.shape


yelp_new.boxplot('sentiment_score', by='stars')


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
from sklearn.grid_search import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()


X = yelp_new['text']
X.head()

y = yelp_new['stars']
y.head()


from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score


# 0.7824
pipe_nb_vect = make_pipeline(vect, nb)
cross_val_score(pipe_nb_vect, X, y, cv=5, scoring='accuracy').mean()


# 0.7985
pipe_logr_vect = make_pipeline(vect, logr)
cross_val_score(pipe_logr_vect, X, y, cv=5, scoring='accuracy').mean()


# 0.6025
pipe_nb_tfidf = make_pipeline(tfidf, nb)
cross_val_score(pipe_nb_tfidf, X, y, cv=5, scoring='accuracy').mean()


# 0.7720
pipe_logr_tfidf = make_pipeline(tfidf, logr)
cross_val_score(pipe_logr_tfidf, X, y, cv=5, scoring='accuracy').mean()


import scipy as sp


# 0.8123, 7min 44s
pipe_nb_vect.named_steps.keys()

nb_vect = {}
nb_vect['countvectorizer__max_df'] = [0.1, 0.2, 0.3, 0.4]
nb_vect['countvectorizer__min_df'] = [1, 2, 3]
nb_vect['countvectorizer__stop_words'] = [None, 'english']
nb_vect['countvectorizer__ngram_range'] = [(1, 2)]
nb_vect['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
nb_vect


rand_nb_vect = RandomizedSearchCV(pipe_nb_vect, nb_vect, cv=5, scoring='accuracy', n_iter=50, random_state=1)

%time rand_nb_vect.fit(X, y)

print(rand_nb_vect.best_score_)
print(rand_nb_vect.best_params_)


# 0.7816, 7min 49s
pipe_nb_tfidf.named_steps.keys()

nb_tfidf = {}
nb_tfidf['tfidfvectorizer__max_df'] = [0.1, 0.2, 0.3, 0.4]
nb_tfidf['tfidfvectorizer__min_df'] = [1, 2, 3]
nb_tfidf['tfidfvectorizer__stop_words'] = [None, 'english']
nb_tfidf['tfidfvectorizer__ngram_range'] = [(1, 2)]
nb_tfidf['tfidfvectorizer__norm'] = ['l1', 'l2', None]
nb_tfidf['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
nb_tfidf

rand_nb_tfidf = RandomizedSearchCV(pipe_nb_tfidf, nb_tfidf, cv=5, scoring='accuracy', n_iter=50, random_state=1)

%time rand_nb_tfidf.fit(X, y)

print(rand_nb_tfidf.best_score_)
print(rand_nb_tfidf.best_params_)


# 0.8069, 12min 28s
pipe_logr_vect.named_steps.keys()

logr_vect = {}
logr_vect['countvectorizer__max_df'] = [0.1, 0.2, 0.3, 0.4]
logr_vect['countvectorizer__min_df'] = [1, 2, 3]
logr_vect['countvectorizer__stop_words'] = [None, 'english']
logr_vect['countvectorizer__ngram_range'] = [(1, 2)]
logr_vect['logisticregression__penalty'] = ['l1', 'l2']
logr_vect['logisticregression__C'] = sp.stats.uniform(scale=1)
logr_vect

rand_logr_vect = RandomizedSearchCV(pipe_logr_vect, logr_vect, cv=5, scoring='accuracy', n_iter=50, random_state=1)

%time rand_logr_vect.fit(X, y)

print(rand_logr_vect.best_score_)
print(rand_logr_vect.best_params_)


# 0.8083, 10min 13s
pipe_logr_tfidf.named_steps.keys()

logr_tfidf = {}
logr_tfidf['tfidfvectorizer__max_df'] = [0.1, 0.2, 0.3, 0.4, 0.5]
logr_tfidf['tfidfvectorizer__min_df'] = [1, 2, 3]
logr_tfidf['tfidfvectorizer__stop_words'] = [None, 'english']
logr_tfidf['tfidfvectorizer__ngram_range'] = [(1, 2)]
logr_tfidf['tfidfvectorizer__norm'] = ['l1', 'l2', None]
logr_tfidf['logisticregression__penalty'] = ['l1', 'l2']
logr_tfidf['logisticregression__C'] = sp.stats.uniform(scale=1)
logr_tfidf

rand_logr_tfidf = RandomizedSearchCV(pipe_logr_tfidf, logr_tfidf, cv=5, scoring='accuracy', n_iter=50, random_state=1)

%time rand_logr_tfidf.fit(X, y)

print(rand_logr_tfidf.best_score_)
print(rand_logr_tfidf.best_params_)


def get_sentiment(df):
    return df['sentiment_score']

get_sentiment(yelp_new).head()
get_sentiment(yelp_new).shape


def get_text(df):
    return df['text']

get_text(yelp_new).head()
get_text(yelp_new).shape


rand_nb_vect.best_estimator_

vect = CountVectorizer(max_df=0.2, min_df=3, ngram_range=(1,2))
nb = MultinomialNB(alpha=0.65297866465495935)


from sklearn.preprocessing import FunctionTransformer

get_text_ft = FunctionTransformer(get_text, validate=False)
get_sentiment_ft = FunctionTransformer(get_sentiment, validate=False)


from sklearn.pipeline import make_union

union = make_union(vect)
X_dtm = union.fit_transform(X)
X_dtm.shape

union = make_union(make_pipeline(get_text_ft, vect), get_sentiment_ft)
X_dtm_new = union.fit_transform(yelp_new)
X_dtm_new.shape





