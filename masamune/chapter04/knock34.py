import ast

nouns = []; word = []
with open('morpheme.txt') as file:
    for sentence in file:
        sentence_dic = ast.literal_eval(sentence) #文字列を辞書に変換
        for morpheme in sentence_dic:
            if morpheme['pos'] == '名詞':
                word.append(morpheme['surface'])
            elif len(word) >= 2:
                nouns.append(''.join(word))
                word = []
    
    print('\n'.join(nouns))