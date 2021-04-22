'''
https://qiita.com/TodayInsane/items/94f495db5ba143a8d3e0
'''

def cipher(sentecne):
    new_sentence = ""
    for character in sentence:
        if character.islower():
            new_sentence += chr(219-ord(character))
        else:
            new_sentence += character
    return new_sentence

sentence = "I am Japanese."
print(cipher(sentence))
