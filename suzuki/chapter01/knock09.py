# スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．
# 適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）
# を与え，その実行結果を確認せよ．

import random

def weird(text):
    words_list = text.split(" ")
    ans = []
    
    for word in words_list:
        if len(word) > 4:
            x = random.sample(list(word[1:-1]),len(word[1:-1])) #両端以外をシャッフル
            weird_word = ''.join(list(word[0]) + x + list(word[-1])) #両端とxをつなげて文字列型に直す
            ans.append(weird_word)
        
        else: 
            ans.append(word)
    
    return ans

Input = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

print(' '.join(weird(Input)))
