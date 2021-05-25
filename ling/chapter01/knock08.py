#08
import sys
def cipher(x):
    r=""
    for w in x:
        if w.islower():
            r+=chr(219-ord(w))
        else:
            r+=w
    return r
x="abcdeABCED"
a=cipher(x)#encode
b=cipher(x)#decode
if a==b:
    print("nice")