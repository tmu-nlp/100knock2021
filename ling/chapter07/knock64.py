from gensim.models import KeyedVectors
model=KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True)

with open("questions-words.txt",'r',encoding='utf-8') as f:
    data=f.readlines()
with open('ans.txt','w') as a:
    for line in data:
        line=line.split()
        if line[0]==':':
            cate=line[1]
            print(cate)
        else:
            word,cos_smi=model.most_similar(positive=[line[1],line[2]],negative=[line[0]],topn=1)[0]
            a.write(cate+' '+' '.join(line+[word,str(cos_smi)+'\n']))


