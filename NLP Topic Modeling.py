# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:54:47 2018

@author: SHAJI JAMES
"""

# =============================================================================
# establishing connection
# =============================================================================
import tweepy
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

#get tweets 
fetched_tweets = api.search(q = '#digitalindia', count = 100,lang='en')

tweets=[]
for tweet in fetched_tweets: 
    if tweet.retweet_count > 0: 
        if tweet.text not in tweets: 
            tweets.append(tweet.text) 
    else: 
        tweets.append(tweet.text)

# =============================================================================
# preprocessing
# =============================================================================
import re
def clean_tweet(tweet):
    return str.lower(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", ' ', tweet))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

refined_tokens=[]
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
for tweet in tweets:
    tokens=word_tokenize(clean_tweet(tweet))
    word_list=[]
    for word in tokens:
        if word not in stop_words:
            word_list.append(lemmatizer.lemmatize(word, pos='v'))
    refined_tokens.append(word_list)

# =============================================================================
# creating word corpus
# =============================================================================
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(refined_tokens)

#visualising dictionary
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#Preparing document term matrix
doc_term_matrix = [dictionary.doc2bow(token) for token in refined_tokens]

#visualising the occurence of terms in a single document
doc_20=doc_term_matrix[20]
for i in range(len(doc_20)):
    print('No: ',doc_20[i][0],'Word: ',dictionary[doc_20[i][0]],'| Occurence(s): ',doc_20[i][1])

# =============================================================================
# model building
# =============================================================================
#transformed corpus
tfidf = gensim.models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

#Running LDA using Bag of Words
ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)
ldamodel.print_topics(num_topics=10, num_words=3)

#Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=50)
lda_model_tfidf.print_topics(num_topics=10, num_words=3)

# =============================================================================
# performance evaluation
# =============================================================================
#evaluating lda bag of words model for a single document
for index, score in ldamodel[doc_term_matrix[20]]:
    print('Score: ',score,'Topic:', ldamodel.print_topic(index, 3))

#evaluating lda tf-idf model for a single document   
for index, score in lda_model_tfidf[doc_term_matrix[20]]:
    print('Score: ',score,'Topic:', lda_model_tfidf.print_topic(index, 3))

# =============================================================================
# testing
# =============================================================================
unseen_document = 'Digital India is a great initiative to begin with. If executed properly, a great rise in the development of India can be seen soon enough.'
test_token=word_tokenize(clean_tweet(unseen_document))
test_word_list=[]
for word in test_token:
    if word not in stop_words:
        test_word_list.append(lemmatizer.lemmatize(word, pos='v'))
test_dictionary = corpora.Dictionary([test_word_list])
bow_vector = test_dictionary.doc2bow(test_word_list)

for index, score in ldamodel[bow_vector]:
    print('Score: ',score,'Topic:', ldamodel.print_topic(index, 3))
