# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:29:42 2017

ColumbiaX CSMM.101x Artificial Intelligence Project 3 Part 3

Usage: python problem3_3.py <input_filename> <output_filename>
<input_filename>: input1.csv, containing x1, x2 input and y label in csv format 
                  with a single row as header
<output_filename>: method, best score, test score in csv format

score: 100/100 past all test cases on Vocareum.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def run():
    """ 
    Read the input CSV file, first 2 columns data, 3rd column label
    """
    target = open('output3.csv', 'w')
    source = np.loadtxt('input3.csv', delimiter=",", skiprows=1)
    X = source[:,:2]
    y = source[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

    " SVM w/ linear kernel "
    param_grid = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}]
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5)
    clf.fit(X_train, y_train)
    print("\nlinear kernel svm:")
    print("best param:",clf.best_params_)
    print("best training accuracy:", clf.best_score_, "\ntest score accuracy", clf.score(X_test, y_test))
    target.write('svm_linear,'+str(clf.best_score_)+','+str(clf.score(X_test,y_test))+'\n')
    
    " SVM w/ polynomial kernel "
    param_grid = [{'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 1], 'kernel': ['poly']}] # hung
    clf = GridSearchCV(SVC(), param_grid, cv=5)
    clf.fit(X_train, y_train)
    print("\npolynomial kernel svm:")
    print("best param:",clf.best_params_)
    print("best training accuracy:", clf.best_score_, "\ntest score accuracy", clf.score(X_test, y_test))
    target.write('svm_polynomial,'+str(clf.best_score_)+','+str(clf.score(X_test,y_test))+'\n')
    
    " SVM w/ rbf kernel "
    param_grid = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10], 'kernel':['rbf']}]
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5)
    clf.fit(X_train, y_train)
    print("\nrbf kernel svm:")
    print("best param:",clf.best_params_)
    print("best training accuracy:", clf.best_score_, "\ntest score accuracy", clf.score(X_test, y_test))
    target.write('svm_rbf,'+str(clf.best_score_)+','+str(clf.score(X_test,y_test))+'\n')

    " SVM w/ logistic kernel "
    param_grid = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel':['sigmoid']}]
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5)
    clf.fit(X_train, y_train)
    print("\nlogistic kernel svm:")
    print("best param:",clf.best_params_)
    print("best training accuracy:", clf.best_score_, "\ntest score accuracy", clf.score(X_test, y_test))
    target.write('svm_logistic,'+str(clf.best_score_)+','+str(clf.score(X_test,y_test))+'\n')
    
    " KNN kernel grid search, result different each time. "
    from sklearn.neighbors import KNeighborsClassifier
    param_grid = [{'n_neighbors': list(range(1,51)), 'leaf_size': list(np.arange(5, 65, 5))}]
    knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    knn.fit(X_train, y_train)
    print("\nKNN grid search:")
    print("best param:",knn.best_params_)
    print("best training accuracy:", knn.best_score_, "\ntest score accuracy", knn.score(X_test, y_test))
    target.write('knn,'+str(knn.best_score_)+','+str(knn.score(X_test,y_test))+'\n')
    
    " decision tree grid search, result different each time. "
    from sklearn.tree import DecisionTreeClassifier    
    param_grid = [{'max_depth': list(range(1,51)), 'min_samples_split': list(range(2,11))}]
    dtc = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5, scoring='accuracy')
    dtc.fit(X_train, y_train)
    print("\nDecision Tree grid search:")
    print("best param:",dtc.best_params_)
    print("best training accuracy:", dtc.best_score_, "\ntest score accuracy", dtc.score(X_test, y_test))
    target.write('decision_tree,'+str(dtc.best_score_)+','+str(dtc.score(X_test,y_test))+'\n')

    " random forest grid search. result different each time. "
    from sklearn.ensemble import RandomForestClassifier
    param_grid = [{'max_depth': list(range(1,51)), 'min_samples_split': list(range(2,11))}]
    rf = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5, scoring='accuracy')
    rf.fit(X_train, y_train)
    print("\nrandom forest grid search:")
    print("best param:",rf.best_params_)
    print("best training accuracy:", rf.best_score_, "\ntest score accuracy", rf.score(X_test, y_test))
    target.write('random_forest,'+str(rf.best_score_)+','+str(rf.score(X_test,y_test))+'\n')
    
    target.close()

if __name__ == '__main__':
    run()