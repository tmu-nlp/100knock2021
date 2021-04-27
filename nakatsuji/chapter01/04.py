from collections import *
S = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = [s for s in S.split(" ")]
a_char2idx = defaultdict()
for i in range(len(words)):
    if i+1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        a_char2idx[words[i][0]] = words.index(words[i]) + 1
    else:
        a_char2idx[words[i][1]] = words.index(words[i]) + 1
out = dict(a_char2idx)
print(out)