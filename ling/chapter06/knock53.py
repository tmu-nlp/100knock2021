import numpy as np
from knock52 import LR
from knock51_pd import X_train,X_test
'''
@methods
predict_proba(X): Probability estimates. The returned estimates for all classes are ordered by the label of classes.
'''
def score_of_model(LR,X):
    return [np.max(LR.predict_proba(X), axis=1), LR.predict(X)]

train_pred=score_of_model(LR,X_train)
test_pred = score_of_model(LR, X_test)

#print(train_pred)