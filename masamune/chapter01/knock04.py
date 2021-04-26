s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
s = s.replace(".", "").split()
num = [1, 5, 6, 7, 8, 9, 15, 16, 19]
dic = {}
for i, v in enumerate(s):
    if i+1 in num:
        v = v[:1]
    else:
        v = v[:2]
    dic[v] = i + 1

print(dic)