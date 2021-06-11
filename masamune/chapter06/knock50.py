import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

with zipfile.ZipFile('NewsAggregatorDataset.zip') as zip_f:
    zip_f.extractall('./data')

#データ読み込み
names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
df = pd.read_csv('./data/newsCorpora.csv', sep='\t', header=None, names=names)

#データ抽出
extract = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
df = df.loc[df['PUBLISHER'].isin(extract), ['TITLE', 'CATEGORY']]

#データ分割
train_data, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=0, stratify=df['CATEGORY'])
valid_data, test_data = train_test_split(test, test_size=0.5, random_state=0, stratify=test['CATEGORY'])

#データ保存
train_data.to_csv('./data/train.txt', sep='\t', index=False, header=None)
valid_data.to_csv('./data/valid.txt', sep='\t', index=False, header=None)
test_data.to_csv('./data/test.txt', sep='\t', index=False, header=None)

#事例数
print(train_data['CATEGORY'].value_counts())
print(valid_data['CATEGORY'].value_counts())
print(test_data['CATEGORY'].value_counts())

"""
b    4502
e    4223
t    1219
m     728
Name: CATEGORY, dtype: int64
b    562
e    528
t    153
m     91
Name: CATEGORY, dtype: int64
b    563
e    528
t    152
m     91
Name: CATEGORY, dtype: int64
"""