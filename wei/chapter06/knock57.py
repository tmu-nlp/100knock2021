""'''
[task description]特徴量の重みの確認
52で学習したロジスティック回帰モデルの中で，
重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．
'''

from knock50 import load_df
from sklearn.model_selection import train_test_split
from knock51 import df_pre, seg
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from IPython import display


def sorted_by_weight(lg):

    for c, coef in zip(lg.classes_, lg.coef_):       # インスタンス名.coef_ で、パラメータ（重み）を取得する
        print(f'【カテゴリ】{c}')
        best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['重要度上位'], index=index).T
        worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T
        display.display(pd.concat([best10, worst10], axis=0))
        print('\n')


if __name__ == '__main__':
    # データの読み込み
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    # データの分割
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    # データを前処理した後、データを再結合
    df_pre = df_pre(train, valid, test)
    X_train = seg(df_pre, train, valid)[0]

    lg = LogisticRegression(random_state=123, max_iter=10000, solver='lbfgs', multi_class='auto')
    lgfit = lg.fit(X_train, train['CATEGORY'])
    features = X_train.columns.values
    # print(features)                             # 2815 features -> ['0m' '0million' '0nd' ... 'zac efron' 'zendaya' 'zone']
    index = [i for i in range(1, 11)]
    sorted_by_weight(lgfit)

'''
output example like:
【カテゴリ】b
          1      2      3    4   ...         7       8       9          10
重要度上位   bank    fed  china  ecb  ...  obamacare     oil  yellen     dollar
重要度下位  video  ebola    the  her  ...      apple  google    star  microsoft

[2 rows x 10 columns]
'''