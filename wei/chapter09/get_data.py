import pandas as pd


base1 = '../chapter06/'
train= pd.read_csv(base1 + 'train.txt', sep='\t', header=None)
valid = pd.read_csv(base1 + 'valid.txt', sep='\t', header=None)
test = pd.read_csv(base1 + 'test.txt', sep='\t', header=None)

d = {'b':0, 't':1, 'e':2, 'm':3}
y_train = train.iloc[:,1].replace(d)
y_train.to_csv('y_train.txt',header=False, index=False)
y_valid = valid.iloc[:,1].replace(d)
y_valid.to_csv('y_valid.txt',header=False, index=False)
y_test = test.iloc[:,1].replace(d)
y_test.to_csv('y_test.txt',header=False, index=False)