from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from knock51_pd import X_train,train,X_valid,valid,X_test,test
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
@para
LR: model
X: train dataset
'''
def score_of_model(LR,X):
    return [np.max(LR.predict_proba(X), axis=1), LR.predict(X)]

result = []


'''
logspace(start,end,number of samples,base)
num of samples=20 time: 08:43
'''
for c in tqdm(np.logspace(-5,4,10,base=10)):
    LR=LogisticRegression(random_state=42,max_iter=10000,C=c)
    LR.fit(X_train,train['CATEGORY'])

    train_pred=score_of_model(LR,X_train)
    valid_pred=score_of_model(LR,X_valid)
    test_pred = score_of_model(LR, X_test)

    train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])
    valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1])
    test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])

    result.append([c, train_accuracy, valid_accuracy, test_accuracy])

result = np.array(result).T
plt.plot(result[0], result[1], label='train')
plt.plot(result[0], result[2], label='valid')
plt.plot(result[0], result[3], label='test')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.xscale ('log')
plt.xlabel('c')
plt.legend()
plt.show()

