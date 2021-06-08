from sklearn.linear_model import LogisticRegression
from knock51_pd import X_train,train

LR = LogisticRegression(random_state=42,max_iter=10000)
LR.fit(X_train, train['CATEGORY'])
#print('done')