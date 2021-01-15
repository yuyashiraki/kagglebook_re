# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# RMSE(Root Mean Squared Error)
from sklearn.metrics import mean_squared_error


y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532


# Confusion Matrix
from sklearn.metrics import confusion_matrix

y_true = [1,0,1,1,0,1,1,0]
y_pred = [0,0,1,1,0,0,1,1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)

confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)


# accuracy
from sklearn.metrics import accuracy_score

y_true = [1,0,1,1,0,1,1,0]
y_pred = [0,0,1,1,0,0,1,1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)


# logloss
from sklearn.metrics import log_loss

y_true = [1,0,1,1,0,1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)


# multi-class logloss
from sklearn.metrics import log_loss
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)

# multi-label f1-score
from sklearn.metrics import f1_score

y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])
mean_f1 = np.mean([f1_score(y_true[i,:], y_pred[i,:]) for i in range(len(y_true))])

n_class = 3
macro_f1 = np.mean([f1_score(y_true[:,c], y_pred[:,c]) for c in range(n_class)])

micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)

mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# quadratic weighted kappaを計算する関数
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom

y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# MAP@K
K = 3
y_true = [[1,2],[1,2],[4],[1,2,3,4],[3,4]]
y_pred = [[1,2,4],[4,1,2],[1,4,3],[1,2,3],[1,2,4]]

'''
average precision for each record
'''
def apk(y_i_true, y_i_pred):
    assert(len(y_i_pred) <= K)
    assert(len(np.unique(y_i_pred)) == len(y_i_pred))
    sum_precision = 0.0
    num_hits = 0.0

    for i,p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision
    return sum_precision / min(len(y_i_true), K)

def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

print(mapk(y_true, y_pred))
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
