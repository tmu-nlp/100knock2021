# 53.py

import numpy as np

def score_lg(lg, X):
  return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]

train_pred = score_lg(lg, X_train)
test_pred = score_lg(lg, X_test)

print(train_pred)