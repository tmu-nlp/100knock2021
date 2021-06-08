import re,string
from knock50_pd import train,test,valid
import pandas as pd
import numpy as np

'''
@para
text
change punctuation into space
change all number string into 0
lower
'''
def preprocess(text):
    '''
    str.maketrans: make a table looks like {a:b,c:d} for string transforming
    '''
    table=str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table)
    text=re.sub('[0-9]+','0',text)
    text=text.lower()
    return text

'''
pd.concat(obj,axis,ignore_index)
@para
obj: a sequence or mapping of Series or DataFrame objects
axis: {0/’index’, 1/’columns’}, default 0
ignore_index: If True, do not use the index values along the concatenation axis
'''
df=pd.concat([train,valid,test],axis=0,ignore_index=True)

'''
Series.map(arg, na_action=None)
Used for substituting each value in a Series with another value, that may be derived from a function, a dict or a Series.
arg: function, collections.abc.Mapping subclass or Series
'''
df['TITLE']=df['TITLE'].map(lambda x : preprocess(x))

#separate df into train set, valid set and test set
train_valid = df[:len(train) + len(valid)]
train = train_valid[:len(train)]
valid = train_valid[len(train):]
test = df[len(train) + len(valid):]

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
'''
TfidfVectorizer(min_df=10,ngram_range=(1,2))
@para
min_df: ignore terms that have a document frequency strictly lower than the given threshold
ngram_range=(1,2): (1, 2) means unigrams and bigrams

@methods
fit_transform(raw_documents[, y]):  Learn vocabulary and idf, return document-term matrix.
transform(raw_documents): Transform documents to document-term matrix.

'''

vectorizer = TfidfVectorizer(min_df=10,ngram_range=(1,2))
X_train = vectorizer.fit_transform(train['TITLE'])
X_valid = vectorizer.transform(valid['TITLE'])
X_test = vectorizer.transform(test['TITLE'])

X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names())
X_valid = pd.DataFrame(X_valid.toarray(), columns=vectorizer.get_feature_names())
X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names())


X_train.to_csv('./train.feature.txt', sep='\t', index=False)
X_valid.to_csv('./valid.feature.txt', sep='\t', index=False)
X_test.to_csv('./test.feature.txt', sep='\t', index=False)

