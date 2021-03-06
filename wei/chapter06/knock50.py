""'''
[task description]データの入手・整形
News Aggregator Data Setを以下の要領で
学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ.
1. 情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する.
2. 抽出された事例をランダムに並び替える.
3. 抽出された事例を, それぞれtrain.txt(80%)，valid.txt(10%)，test.txt(10%)というファイル名で保存する．
ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ.

[file_1 description]
filename: newsCorpora.csv
format(8 items)
ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
where:
PUBLISHER	Publisher name
CATEGORY	News category
            (b = business, t = science and technology,
            e= entertainment, m = health)
STORY		Alphanumeric ID of the cluster that includes news about the same story

'''
import pandas as pd
from sklearn.model_selection import train_test_split     # data分割


def load_df(infile):
    df = pd.read_csv(infile, header=None, sep='\t',
                     names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    # print(df.info())
    # データの抽出　->  .loc(x_label, y_label)で索引
    df = df.loc[df['PUBLISHER'].astype(str).isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']),
            ['TITLE', 'CATEGORY']]
    return df


if __name__ == '__main__':
    '''
    データの読み込み,行数：422937 ./newsCorpora.csv
    読込時のエラー回避のためダブルクォーテーションをシングルクォーテーションに置換:
    use $ sed -e 's/"/'\''/g' ./newsCorpora.csv > ./newsCorpora_re.csv to make 'newsCorpora_re.csv',
    '''
    infile = './data/NewsAggregatorDataset/newsCorpora_re.csv'
    df = load_df(infile)
    '''
    データの分割
    stratify Objectを利用して、指定したカラムの構成比が分割後の各データで等しくなるように分割される.
    分類の目的変数であるCATEGORYを指定し、データごとに偏りないようにする.
    '''
    train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123,
                                   stratify=valid_test['CATEGORY'])
    # データの保存
    train.to_csv('./train.txt', columns=['TITLE', 'CATEGORY'], sep='\t', header=False, index=False)
    valid.to_csv('./valid.txt', columns=['TITLE', 'CATEGORY'], sep='\t', header=False, index=False)
    test.to_csv('./test.txt', columns=['TITLE', 'CATEGORY'], sep='\t', header=False, index=False)

    # 事例数の確認
    print('[train_data]')
    print(train['CATEGORY'].value_counts())
    print('[valid_data]')
    print(valid['CATEGORY'].value_counts())
    print('[test_data]')
    print(test['CATEGORY'].value_counts())

'''
[train_data]
b    4501
e    4235
t    1220
m     728
Name: CATEGORY, dtype: int64
[valid_data]
b    563
e    529
t    153
m     91
Name: CATEGORY, dtype: int64
[test_data]
b    563
e    530
t    152
m     91
Name: CATEGORY, dtype: int64
'''


