#06 chara-bigram
import sys

def bigram(In):
    g=[]
    for i in range(len(In)-2+1):
        g.append(In[i:i+2])
    return g
word1,word2="paraparaparadise","paragraph"
X,Y=bigram(word1),bigram(word2)

print("[06]. \nunion="+str(set(X)|set(Y))+"intersection: "+str(set(X)&set(Y))+ "difference: "+str(set(X)^set(Y))+"\n")