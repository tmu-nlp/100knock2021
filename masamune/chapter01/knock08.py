def cipher(message):
    result = ''
    for i in range(len(message)):
        if message[i].islower():
            result += chr(219 - ord(message[i]))
        else:
            result += message[i]
            
    return result

s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
print("暗号化 : " + cipher(s))
print("複合化 : " + cipher(cipher(s)))