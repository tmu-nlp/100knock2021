'''
Task:
1.学習データ，検証データ，評価データを行列・ベクトルに変換する
2.すべての事例xiの特徴ベクトルxiを並べた行列Xと正解ラベルを並べた行列（ベクトル）Yを作成
3.「ビジネス」「科学技術」「エンターテイメント」「健康」の4カテゴリ分類、ラベルは「０、１、２、３」にする
'''
import string,torch,re

from pandas.core.indexes import category
import pandas as pd
from gensim.models import KeyedVectors

#pretrained word vector model
model=KeyedVectors.load_word2vec_format('/Users/lingzhidong/Documents/GitHub/100knock2021/ling/chapter07/GoogleNews-vectors-negative300.bin',binary=True)

#read train set,valid set,and test set
train=pd.read_csv('./_train.txt',header=0,sep='\t',names=['TITLE','CATEGORY'])
valid=pd.read_csv('./_valid.txt',header=0,sep='\t',names=['TITLE','CATEGORY'])
test=pd.read_csv('./_test.txt',header=0,sep='\t',names=['TITLE','CATEGORY'])

'''
print('【学習データ】')
print(train['CATEGORY'].value_counts())
print('【検証データ】')
print(valid['CATEGORY'].value_counts())
print('【評価データ】')
print(test['CATEGORY'].value_counts())
'''
#preprocessing of the word
'''
remove the punctuation, take vectors of each word in title, 
return the average as the vector of the instance 
'''
def preprocess(text):
    
    table=str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table).split()
    vec=[model[word] for word in text if word in model]
    vec_mean=torch.tensor(sum(vec)/len(vec))
    return vec_mean

#create feature vector for instance
X_train=torch.stack([preprocess(text) for text in train['TITLE']])
X_valid=torch.stack([preprocess(text) for text in valid['TITLE']])
X_test=torch.stack([preprocess(text) for text in test['TITLE']])

'''
print(X_train.size())
'''
#create label vector for each instance
category_dict={'b':0,'t':1,'e':2,'m':3}
y_train=torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid=torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test=torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

'''
print(y_train)
'''

torch.save(X_train,'X_train.pt')
torch.save(X_valid,'X_valid.pt')
torch.save(X_test,'X_test.pt')
torch.save(y_train,'y_train.pt')
torch.save(y_valid,'y_valid.pt')
torch.save(y_test,'y_test.pt')

