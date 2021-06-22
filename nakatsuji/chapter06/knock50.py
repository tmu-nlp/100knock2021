import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/drive/MyDrive/nlp100/newsCorpora.csv', header=None, sep='\t', names =['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
df = df.loc[df['PUBLISHER'].isin(publishers)]
df = df.loc[:, ['TITLE', 'CATEGORY']]
df.head()
train, valid_test = train_test_split(df, train_size=0.8, shuffle=True, random_state=50, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=50, stratify=valid_test['CATEGORY'])

path = '/content/drive/MyDrive/nlp100'
train.to_csv(path + '/train.txt', sep='\t', index=False)
valid.to_csv(path + '/valid.txt', sep='\t', index=False)
test.to_csv(path + '/test.txt', sep='\t', index=False)

print('train')
print(train['CATEGORY'].value_counts())
print('valid')
print(valid['CATEGORY'].value_counts())
print('test')
print(test['CATEGORY'].value_counts())
