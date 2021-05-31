import ast

noun_phrase = set()
with open('morpheme.txt') as file:
    for sentence in file:
        sentence_dic = ast.literal_eval(sentence) #文字列を辞書に変換
        for i in range(1, len(sentence_dic)-1):
            if sentence_dic[i]['surface'] == 'の':
                if sentence_dic[i-1]['pos'] == '名詞' and sentence_dic[i+1]['pos'] == '名詞':
                    noun_phrase.add(sentence_dic[i-1]['surface'] + sentence_dic[i]['surface'] + sentence_dic[i+1]['surface'])

    for i in range(10):
        print(list(noun_phrase)[i])