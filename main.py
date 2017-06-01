# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:05:21 2017

"""
import sys

from code.nlp import load_data
from code.features import preprocess
from code.labels import load_labels
from code.model import train, grid_search
from code.model_test import test

print('loading data')
df, dft = load_data('--quick' in sys.argv)

print('preprocessing training set')
preprocess(df)
X_train = df['tokens']

print('generating training labels')
Y_train = load_labels(df)

if '--gridsearch' in sys.argv:
    print('gridsearching')
    grid_search(X_train, Y_train)
    exit()

print('training')
model = train(X_train, Y_train)

if '--test' in sys.argv:
    print('preprocessing test set')
    preprocess(dft)

    print('labelling test data')
    X_test = dft['tokens']
    test(model, X_train, Y_train, X_test)
