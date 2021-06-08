from sklearn.metrics import confusion_matrix
from knock52 import train
def knock55():
    lr, data, vectorizer = train()
    print(confusion_matrix(y_true = data[0][1], y_pred = lr.predict(data[0][0])))
    print(confusion_matrix(y_true = data[1][1], y_pred = lr.predict(data[1][0])))

if __name__ == "__main__":
    knock55()