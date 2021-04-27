def cipher(S):
    S = [chr(219 - ord(w)) if 97 <= ord(w) <= 122 else w for w in S]
    return ''.join(text)
S = "this is a message."
ans = cipher(S)
print(ans)
ans = cipher(ans)
print(ans)