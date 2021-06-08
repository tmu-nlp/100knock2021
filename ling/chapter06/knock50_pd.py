import pandas as pd
from sklearn.model_selection import train_test_split
'''
pd.read_csv(header,names,sep)
@para
header : [int] Row number(s) to use as the column names, and the start of the data
names : [list] List of column names to use
sep : [str] Delimiter
'''

df=pd.read_csv('./NewsAggregatorDataset/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
#print(df)

'''
df.loc(condition,columns)
Extract 2 columns of the line which the publisher is one of the elements in the list
'''

df=df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]



'''
train_test_split(arrays,test_size=None,shuffle=True,random_state=None,stratify=None)
@para
test_size:If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
shuffle:Shuffle the data before splitting
random_state: If this is not None,no matter how many times of shuffle, the result will be the same
stratify: If not None, data is split in a stratified fashion, using this as the class labels.
'''
train,valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])


train.to_csv('./_train.txt', sep='\t', index=False)
valid.to_csv('./_valid.txt', sep='\t', index=False)
test.to_csv('./_test.txt', sep='\t', index=False)

'''
print('学習データ')
print(train['CATEGORY'].value_counts())
print('検証データ')
print(valid['CATEGORY'].value_counts())
print('評価データ')
print(test['CATEGORY'].value_counts())
'''