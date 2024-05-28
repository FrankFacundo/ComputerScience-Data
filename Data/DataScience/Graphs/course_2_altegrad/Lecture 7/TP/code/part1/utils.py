"""
Learning on Sets - ALTEGRAD - Jan 2021
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    X_train = np.zeros(shape=(n_train,max_train_card))
    cards = np.random.randint(1,max_train_card+1,size=n_train)
    for i,n in enumerate(cards):
        X_train[i,-n:] = np.random.randint(1,11,size=n)
    y_train = X_train.sum(axis=1) 
    ##################

    return X_train, y_train


def create_test_dataset():
	
    ############## Task 2
    
    ##################
    # your code here #
    X_test = []
    y_test = []

    for i in range(5,101,5):
        X_test.append(np.random.randint(1,11,size=(10000,i)))
        y_test.append(X_test[-1].sum(axis=1))    
    ##################

    return X_test, y_test