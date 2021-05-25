import ast

verbs = set()
with open('morpheme.txt') as file:
    for sentence in file:
        sentence_dic = ast.literal_eval(sentence) #文字列を辞書に変換
        for morpheme in sentence_dic:
            if morpheme['pos'] == '動詞':
                verbs.add(morpheme['surface'])
    
    for verb in verbs:
        print(verb)