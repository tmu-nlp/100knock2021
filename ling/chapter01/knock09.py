#09
import random,sys
ran="I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
def rdmsw(x):
    r=""
    List=x.split(" ")
    for w in List:
        if len(w)>4:
            l=w[1:len(w)-1]
            tmp=list(l)
            random.shuffle(tmp)
            tmp="".join(tmp)
            l1=""
            l1+=w[0]
            l1+=tmp
            l1+=w[-1]
            r+=l1
            r+=" "
        else:
            r+=w
            r+=" "
    return r

print(rdmsw(ran))