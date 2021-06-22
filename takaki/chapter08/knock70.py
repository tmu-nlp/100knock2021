import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import string
import torch

# --- knock50 ------------------------------------------------------------------

df = pd.read_csv('./tmp/newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]

train, valid_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=123, stratify=df['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=valid_test['CATEGORY'])

# --- knock70 ------------------------------------------------------------------

model = KeyedVectors.load_word2vec_format('./tmp/GoogleNews-vectors-negative300.bin.gz', binary=True)
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}

def transform(text):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  words = text.translate(table).split()
  vec = [model[word] for word in words if word in model]
  return torch.tensor(sum(vec) / len(vec))

def genX(list):
    return torch.stack([transform(text) for text in list['TITLE']])

def genY(list):
    return torch.tensor(list['CATEGORY'].map(lambda x: category_dict[x]).values)

x_train = genX(train)
x_valid = genX(valid)
x_test  = genX(test)

y_train = genY(train)
y_valid = genY(valid)
y_test  = genY(test)
