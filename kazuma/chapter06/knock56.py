from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from knock52 import train
from pprint import pprint
def knock56():
    lr, data, vectorizer = train()
    
    # 一行目がprecision 、二行目がrecall　、、、
    # 一列目がカテゴリ1、二行目がカテゴリ2 、、、
    pprint(precision_recall_fscore_support(y_true = data[1][1], y_pred = lr.predict(data[1][0])))

    # precision, recall, f-score, support の順番
    print("macro:",precision_recall_fscore_support(y_true = data[1][1], y_pred = lr.predict(data[1][0]), average="macro"))
    print("micro:",precision_recall_fscore_support(y_true = data[1][1], y_pred = lr.predict(data[1][0]), average="micro"))

if __name__ == "__main__":
    knock56()