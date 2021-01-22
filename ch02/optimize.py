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

from sklearn.metrics import f1_score
from scipy.optimize import minimize

rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000) # returns evenly spaced numbers

train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)

# clip: limit the val in interval,
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_threshold, init_score)

def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)

# minimize in scipy.optimize
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_threshold, best_score)
