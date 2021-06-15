# 類似度の高い単語10件
# “United States”とコサイン類似度が高い10語と，その類似度を出力せよ．

from gensim.models import KeyedVectors

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('/users/kcnco/github/100knock2021/pan/chapter07/GoogleNews-vectors-negative300.bin', binary = True)
    # "United_States"とコサイン類似度が高い10語を求める
    cos_top10 = model.most_similar('United_States', topn = 10)

    for word, sim in cos_top10:
        print(f'{word}\t{sim}')