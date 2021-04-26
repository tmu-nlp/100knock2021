##Implement a function cipher that converts a given string with the specification:

##Every alphabetical lowercase letter c is converted to a letter whose ASCII code is (219 - [the ASCII code of c])
##Keep other letters unchanged
##Use this function to cipher and decipher an English message.

def encode(s):
    return ''.join([chr(219 - ord(c)) if 'a' <= c <= 'z' else c for c in s])


if __name__ == '__main__':
    s = 'I am a NLPer.'
    s_encode = encode(s)
    s_decode = encode(s_encode)

    print('Original:', s)
    print('Encode:', s_encode)
    print('Decode:', s_decode)
