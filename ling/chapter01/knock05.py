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