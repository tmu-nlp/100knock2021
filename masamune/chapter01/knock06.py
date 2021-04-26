from knock05 import n_gram

s1 = "paraparaparadise"
s2 = "paragraph"
X = set(n_gram(s1, 2)[0])
Y = set(n_gram(s2, 2)[0])

print(f"和集合 : {X | Y}")
print(f"積集合 : {X & Y}")
print(f"差集合 : {X - Y}")

if "se" in X:
    print("'se' is in X.")
else:
    print("'se' is not in X.")
    
if "se" in Y:
    print("'se' is in Y.")
else:
    print("'se' is not in Y.")