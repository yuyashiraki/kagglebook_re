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

from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# generate sample data
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

thresholds = []
scores_tr = []
scores_va = []

kf = KFold(n_splits=4, random_state=71, shuffle=True)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)):
    tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # obj func
    def f1_opt(x):
        return -f1_score(tr_y, tr_pred_prob >= x)

    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    threshold = result['x'].item()
    score_tr = f1_score(tr_y, tr_pred_prob >= threshold)
    score_va = f1_score(va_y, va_pred_prob >= threshold)
    print(threshold, score_tr, score_va)

    thresholds.append(threshold)
    scores_tr.append(score_tr)
    scores_va.append(score_va)

threshold_test = np.mean(thresholds)
print(threshold_test)
