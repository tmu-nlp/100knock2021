from knock52 import train
from pprint import pprint
import numpy as np
import heapq
def knock57():
    lr, data, vectorizer = train()
    inverse_vectorizer_vocabulary_ = {v: k for k, v in vectorizer.vocabulary_.items()}
    for cnt, class_name in enumerate(lr.classes_):
        lr.coef_[cnt]
        print(class_name)
        for i in heapq.nlargest(10, lr.coef_[cnt]):
            index1 = np.where(lr.coef_[cnt] == i)
            print(inverse_vectorizer_vocabulary_[index1[0][0]],":",i)
        for i in heapq.nsmallest(10, lr.coef_[cnt]):
            index1 = np.where(lr.coef_[cnt] == i)
            print(inverse_vectorizer_vocabulary_[index1[0][0]],":",i)
        print()

if __name__ == "__main__":
    knock57()