sent = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
ind = [1, 5, 6, 7, 8, 9, 15, 16, 19]
ret = {i: s[:1 if i + 1 in ind else 2] for i, s in enumerate(sent.split())}
print(ret)
