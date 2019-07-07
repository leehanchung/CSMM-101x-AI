# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:22:36 2017

ColumbiaX CSMM.101x Artificial Intelligence Project 5

Usage:
python driver_3.py 

Input:
    Stanford Large Movie Review Dataset v1.0

Output:
    Prediction results of different feature extraction models
    unigram.output.txt       -> score 50/50
    bigram.output.txt        -> score 50/50
    unigramtfidf.output.txt  -> score 50/50
    bigramtfidf.output.txt   -> score 50/50

Unigram traning accuracy: 0.93008
Bigram training accuracy: 0.97292
TF-IDF Unigram training accuracy: 0.91684
TF-IDF Bigram training accuracy: 0.93688
"""

""" changed the ../ to ./ to make the path work under Windows environment """
train_path = "./resource/asnlib/public/aclImdb/train/" # use terminal to ls files under this directory
test_path = "./resource/asnlib/public/imdb_te.csv" # test data for grade evaluation

import glob
from string import punctuation
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

""" preload stop words into stop_words list """
stop_words = []
with open("stopwords.en.txt") as file:
    for line in file:
        word = line.strip()
        stop_words.append(word)
    
def remove_stop_words(text):
    translator = str.maketrans('', '', punctuation)
    query_words = text.translate(translator).split()
    result = [word for word in query_words if word not in stop_words]
    return ' '.join(result)

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
    pos_text = glob.glob(inpath+"pos/*.txt")
    neg_text = glob.glob(inpath+"neg/*.txt")
    
    count = 0
    with open(outpath + name, "w", encoding="utf8") as outfile:
        for file in pos_text:
            with open(file, "r", encoding="utf8") as infile:
                text = remove_stop_words(infile.read().lower())
                outfile.write(str(count)+","+text+",1\n")
            count += 1

        for file in neg_text:
            with open(file, "r", encoding="utf8") as infile:
                text = remove_stop_words(infile.read().lower())
                outfile.write(str(count)+","+text+",0\n")
            count += 1
    outfile.close()
    infile.close()
    data = pd.read_csv(outpath+name, header=None)
    return data

if __name__ == "__main__":
    df = imdb_data_preprocess(train_path)
    X_train = df.iloc[:, 1].as_matrix() # training data
    y_train = df.iloc[:,2].as_matrix() # training label
    
    ''' load test data '''
    test_df = pd.read_csv(test_path, encoding='ISO-8859-1', skiprows=0)
    X_test = test_df.iloc[:,1].as_matrix()
    
    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''    
    unigram_vectorizer = CountVectorizer(min_df=1)
    X_train_uni = unigram_vectorizer.fit_transform(X_train)
    """
    Initial score, 35.76/50, so ran GridSearchCV.
    >> parameters = {'alpha':[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], 'penalty':('l2','l1','elasticnet'), 'loss':('hinge', 'log', 'modified_huber', 'squared_hinge')}
    >> sdc = SGDClassifier()
    >> clf = GridSearchCV(sdc, parameters)
    >> clf.fit(X_train_uni, y_train)
    >> clf.best_params_
    {'alpha': 0.003, 'loss': 'log', 'penalty': 'l2'}
    """
    clf1 = SGDClassifier(loss="log", penalty="l2", alpha=0.003).fit(X_train_uni, y_train)
    predict_train1 = clf1.predict(X_train_uni)
    print("Unigram traning accuracy:", accuracy_score(y_train, predict_train1, normalize=True))
    
    X_test_uni = unigram_vectorizer.transform(X_test)
    predict_uni = clf1.predict(X_test_uni)
    with open("unigram.output.txt", "w") as outfile:
        for predict in predict_uni:
            outfile.write(str(predict)+"\n")
    outfile.close()
  	
    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                        token_pattern=r'\b\w+\b', min_df=1)
    X_train_bi = bigram_vectorizer.fit_transform(X_train)
    """clf2.best_params_
    {'alpha': 0.003, 'loss': 'log', 'penalty': 'l2'}"""
    clf2 = SGDClassifier(loss="log", penalty="l2", alpha=0.003).fit(X_train_bi, y_train)
    predict_train2 = clf2.predict(X_train_bi)
    print("Bigram training accuracy:", accuracy_score(y_train, predict_train2, normalize=True))
    
    X_test_bi = bigram_vectorizer.transform(X_test)
    predict_bi = clf2.predict(X_test_bi)
    with open("bigram.output.txt", "w") as outfile:
        for predict in predict_bi:
            outfile.write(str(predict)+"\n")
    outfile.close()
    
    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigram.output.txt'''
    unitfidf_vectorizer = TfidfVectorizer(min_df=1)
    X_train_unitfidf = unitfidf_vectorizer.fit_transform(X_train)
    """clf3.best_params_
    {'alpha': 0.0003, 'loss': 'squared_hinge', 'penalty': 'elasticnet'}"""
    clf3 = SGDClassifier(loss="squared_hinge", penalty="elasticnet", alpha=0.0003).fit(X_train_unitfidf, y_train)
    predict_train3 = clf3.predict(X_train_unitfidf)
    print("TF-IDF Unigram accuracy:", accuracy_score(y_train, predict_train3, normalize=True))
    
    X_test_unitfidf = unitfidf_vectorizer.transform(X_test)
    predict_unitfidf = clf3.predict(X_test_unitfidf)
    with open("unigramtfidf.output.txt", "w") as outfile:
        for predict in predict_unitfidf:
            outfile.write(str(predict)+"\n")
    outfile.close()
    
    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigram.output.txt'''
    bitfidf_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X_train_bitfidf = bitfidf_vectorizer.fit_transform(X_train)
    """clf4.best_params_
    {'alpha': 0.0001, 'loss': 'squared_hinge', 'penalty': 'elasticnet'}"""
    clf4 = SGDClassifier(loss="squared_hinge", penalty="elasticnet", alpha=0.001).fit(X_train_bitfidf, y_train)
    predict_train4 = clf4.predict(X_train_bitfidf)
    print("TF-IDF Bigram accuracy:", accuracy_score(y_train, predict_train4, normalize=True))
    
    X_test_bitfidf = bitfidf_vectorizer.transform(X_test)
    predict_bitfidf = clf4.predict(X_test_bitfidf)
    with open("bigramtfidf.output.txt", "w") as outfile:
        for predict in predict_bitfidf:
            outfile.write(str(predict)+"\n")
    outfile.close()
    pass