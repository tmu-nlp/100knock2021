'''
FILENAME #1: newsCorpora.csv (102.297.000 bytes)
DESCRIPTION: News pages
FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP

FILENAME #2: 2pageSessions.csv (3.049.986 bytes)
DESCRIPTION: 2-page sessions
FORMAT: STORY \t HOSTNAME \t CATEGORY \t URL
'''

import sys,random,numpy
path="./NewsAggregatorDataset/"
file_name="newsCorpora.csv"



with open(path+file_name,'r',encoding='utf-8') as f:
    data=[]
    for line in f:
        l=line.strip().split('\t')
        if l[3]=='Reuters' or l[3]=='Huffington Post' or l[3]=='Businessweek' or l[3]=='Contactmusic.com' or l[3]=='Daily Mail':
            data.append(line)

    random.shuffle(data)
    s=len(data)
    print(s)
    size_of_test=int(0.1*s)
    size_of_valid=int(0.1*s)
    size_of_train=s - size_of_test - size_of_valid

    with open('train.txt','w',encoding='utf-8') as t:
        for i in range(size_of_train):
            t.write(data[i])
    with open('valid.txt','w',encoding='utf-8') as v:
        for i in range(size_of_train,size_of_train+size_of_valid):
            v.write(data[i])
    with open('test.txt','w',encoding='utf-8') as te:
        for i in range(size_of_train+size_of_valid,s):
            te.write(data[i])



    
