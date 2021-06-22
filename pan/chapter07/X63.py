# 63. 加法構成性によるアナロジー
# “Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．

from gensim.models import KeyedVectors

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)
    # "Spain" - "Athens" + "Madrid"
    cos_top10 = model.most_similar(positive = ['Spain', 'Athens'], negative = ['Madrid'], topn = 10)

    # 結果を表示する
    for word, sim in cos_top10:
        print(f'{word}\t{sim}'')