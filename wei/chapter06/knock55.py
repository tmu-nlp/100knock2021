""'''
[task description]混同行列の作成
52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，
学習データおよび評価データ上で作成せよ．
'''


from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, seg
from knock53 import score_lg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    df_pre = df_pre(train, valid, test)
    X_train = seg(df_pre, train, valid)[0]
    X_test = seg(df_pre, train, valid)[2]
    lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')

    lgfitT = lg.fit(X_train, train['CATEGORY'])
    train_pred = score_lg(lgfitT, X_train)
    train_cm = confusion_matrix(train['CATEGORY'], train_pred[1])
    print(train_cm)
    seaborn.heatmap(train_cm, annot=True, cmap='Blues')
    plt.show()


    lgfitTe = lg.fit(X_test, test['CATEGORY'])
    test_pred = score_lg(lgfitTe, X_test)
    test_cm = confusion_matrix(test['CATEGORY'], test_pred[1])
    print(test_cm)
    seaborn.heatmap(test_cm, annot=True, cmap='Reds')
    plt.show()

'''
train_cm
[[4344   93    8   56]
 [  52 4173    2    8]
 [  96  125  494   13]
 [ 192  133    7  888]]
 test_cm
[[556   6   0   1]
 [  3 527   0   0]
 [ 20  37  34   0]
 [ 52  23   0  77]]
'''