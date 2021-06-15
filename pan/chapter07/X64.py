# 64. アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
# 求めた単語と類似度は，各事例の末尾に追記せよ．

from gensim.models import KeyedVectors
from tqdm import tqdm

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)

    with open('/users/kcnco/github/100knock2021/pan/chapter07/questions-words.txt', 'r') as input_file, open('/users/kcnco/github/100knock2021/pan/chapter07/result.txt', 'w') as output_file:
        for line in tqdm(input_file):
            if line[0] == ':':
                # : capital-common-countries のような行はとばす
                print(line.strip(), file = output_file)
                continue

            words = line.strip().split()

            # vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)
            # トップN個の(単語, 類似度)というタプルのリストが返ってくる
            cos_sim = model.most_similar(positive = [words[1], words[2]], negative = [words[0]], topn = 1)[0]

            # 各事例の末尾に付け足す
            words += [cos_sim[0], str(cos_sim[1])]

            # 結果を表示する
            print(' '.join(words), file=output_file)