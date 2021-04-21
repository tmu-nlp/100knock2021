#00.reversed string
string1="stressed"
print("[00]. "+ string1[::-1] + "\n")

#01
string2="パタトクカシーー"
print("[01]. "+ string2[0:len(string2):2] + "\n")

#02
string3="パトカー"
string4="タクシー"
s=""
for i in range(4):
    s+=string3[i]
    s+=string4[i]
print("[02]. "+s+ "\n")

#03
string5="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
s5=string5.replace(",","")
s5=string5.replace(".","")
l=sorted(s5.lower().split())
print("[03]. "+str(l)+ "\n")

#04
string6="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
s=string6.split()
dict04=dict()
    #take 1 character
dict04[s[0][0]]=1
for i in range(4,9):
    dict04[s[i][0]]=i+1
dict04[s[14][0]]=15
dict04[s[15][0]]=16
dict04[s[18][0]]=19
    #take 2 characters
for i in range(1,4):
    dict04[s[i][0:2]]=i+1
for i in range(9,14):
    dict04[s[i][0:2]]=i+1
dict04[s[16][0:2]]=17
dict04[s[17][0:2]]=18
dict04[s[19][0:2]]=20
print("[04]. "+str(dict04)+ "\n")

#05 word-n-gram
#string7=["1","2","3","4","5"]
string7="I am an NLPer."
def ngram(In,n):
    if type(In) is str:
        s=In.replace(".","").split(" ")
    else:
        s=In
    g=[]
    for i in range(len(s)-n+1):
        g.append(s[i:i+n])
    return g
print("[05]. "+str(ngram(string7,2))+ "\n")

#06 chara-bigram
def bigram(In):
    g=[]
    for i in range(len(In)-2+1):
        g.append(In[i:i+2])
    return g
word1,word2="paraparaparadise","paragraph"
X,Y=bigram(word1),bigram(word2)

print("[06]. \nunion="+str(set(X)|set(Y))+"intersection: "+str(set(X)&set(Y))+ "difference: "+str(set(X)^set(Y))+"\n")
#07
#08
#09
#10
    