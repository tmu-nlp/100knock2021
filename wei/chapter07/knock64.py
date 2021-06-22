""'''
[description]アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
求めた単語と類似度は，各事例の末尾に追記せよ.

[data format]
$ wget http://download.tensorflow.org/data/question-words.txt
$ head -5 question-words.txt    ->意味的アナロジーを評価するための組
: capital-common-countries
Athens Greece Baghdad Iraq
Athens Greece Bangkok Thailand
Athens Greece Beijing China
Athens Greece Berlin Germany
$ tail -5 question-words.txt　　->文法的アナロジーを評価するための組
write writes talk talks
write writes think thinks
write writes vanish vanishes
write writes walk walks
write writes work works
'''
import warnings
import gensim


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
filepath = './data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

# 1行ずつ読み込み、指定の単語と類似度を計算した上で、整形したデータを出力
with open('./data/questions-words.txt','r',encoding='utf-8') as f1,\
        open('./knock64_add_sim.txt', 'w', encoding='utf-8') as f2:
    for line in f1:
        line = line.split()
        if line[0] == ':':
            category = line[1]
        else:
            word, sim = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
            f2.write(' '.join([category] + line + [word, str(sim) + '\n']))

