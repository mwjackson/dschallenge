# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:47:23 2017

"""
import numpy as np


def test(model, X_train, Y_train, X_test):

    model.fit(X_train, Y_train)

    Y_test = model.predict(X_test)

    np.savetxt('sub.csv', Y_test, delimiter=',')
