import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./tmp/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]

train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])

train.to_csv('./tmp/train.txt', sep='\t', index=False)
valid.to_csv('./tmp/valid.txt', sep='\t', index=False)
test.to_csv('./tmp/test.txt', sep='\t', index=False)
