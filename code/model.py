# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:45:13 2017

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from operator import itemgetter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import KernelPCA, TruncatedSVD


# pickle can't work with lambdas so use plain defs
def tok(x):
    return x


def prep(x):
    return x


def create_model():
    vectorizer = TfidfVectorizer(tokenizer=tok, preprocessor=prep,
                                 ngram_range=(1, 3), min_df=2)  # max_features=10000

    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(LinearSVC(random_state=42, loss='hinge', C=10), n_jobs=-1))

        # ('classifier',OneVsRestClassifier(SGDClassifier(loss='modified_huber', penalty='l1', n_iter=25), n_jobs=-1))
        # ('classifier', OneVsRestClassifier(VotingClassifier(estimators=[
        #		('sgd', SGDClassifier(loss='modified_huber', penalty='l1', n_iter=25)),
        #		('svc', LinearSVC(C=100)),
        #		('mnb', MultinomialNB())
        #	], voting='soft'), n_jobs=-1))
        # ('classifier', OneVsRestClassifier(AdaBoostClassifier(random_state=42), n_jobs=-1))
    ])

    # model = Pipeline([
    #     ('vectorizer', vectorizer),
    #     ('decom', TruncatedSVD(n_components=50)),
    #     ('scale', MaxAbsScaler()),
    #     ('classifier', MLPClassifier(verbose=1, max_iter=50, random_state=42))
    # ])

    return model


def train(X_train, Y_train):

    docs_train, docs_test, labels_train, labels_test = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42)

    model = create_model()

    # train
    model.fit(docs_train, labels_train)  # mod to use grid search or "model" for the pipeline only

    labels_predict = model.predict(docs_test)

    print(classification_report(labels_test, labels_predict))
    print(accuracy_score(labels_test, labels_predict))

    return model


def grid_search(X_train, Y_train):
    model = create_model()

    # get param names
    print(sorted(model.get_params()))

    # gridsearch params
    parameters = [
        # {'estimator__classifier__C':[0.1,1,10,100],
        # 'estimator__feats__k':[50, 100, 500, 1000],
        # ,{'vectorizer__max_df':[0.5,0.4,0.3]}
        # ,{'vectorizer__min_df':[3,4,5]}
        # {'vectorizer__ngram_range':[(1,1),(1,2),(1,3)]},
        # {'vectorizer__max_features':[10000, 25000, 50000, 100000,150000]}
        {'classifier__estimator__C': [1, 10, 100, 1000], 'classifier__estimator__loss': ['hinge', 'squared_hinge']}
    ]

    gs = GridSearchCV(estimator=model, param_grid=parameters, cv=2, scoring='f1_micro')
    gs.fit(X_train, Y_train)  # mod to use grid search or "model" for the pipeline only

    print(gs.grid_scores_)
    print(gs.cv_results_)
    print(gs.best_params_)



# http://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
# can't use this with one vs rest classifier
def show_most_informative_features(pipeline, text=None, n=10):
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(classifier.coef_[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    # Get the top n and bottom n coef, name pairs
    topn = zip(coefs[:n], coefs[:-(n + 1):-1])

    return list(topn)

# show_most_informative_features(model)
