def cipher(s):
    return ''.join([chr(219 - ord(a)) if a.islower() else a for a in s])
s = input()
ret = cipher(s)
print(ret)
print(cipher(ret))
