import pandas as pd
from sklearn.model_selection import train_test_split
from functools import reduce

news_corpora = pd.read_csv('data/newsCorpora.csv',sep='\t',header=None)
news_corpora.columns = ['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP']

publisher = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
ls_is_specified = [news_corpora.PUBLISHER == p for p in publisher]
is_specified =reduce(lambda a, b: a | b, ls_is_specified)
df = news_corpora[is_specified]
#  3. 並び替え
df = df.sample(frac=1) # 全てをサンプリングするので、並び替えと等価
# 4.保存
train_df, valid_test_df = train_test_split(df, test_size=0.2) # 8:2
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5) # 8:1:1
train_df.to_csv('data/train.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
valid_df.to_csv('data/valid.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test_df.to_csv('data/test.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
#  事例数の確認
df['CATEGORY'].value_counts()