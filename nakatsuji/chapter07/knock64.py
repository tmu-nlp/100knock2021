'''
txt append sim(col2 - col1 + col3)
'''
import pickle
import sys
from tqdm import tqdm
import gensim
#モデル読み込み
#model_file = 'model.sav'
#loaded_model = pickle.load(open(model_file, 'rb'))


if __name__ == '__main__':
    file = 'questions-words.txt'
    with open(file) as target_file, open('most1_sim.txt', 'w') as f:
        for line in tqdm(target_file):
            if line[0] == ':':
                #category名のところは飛ばす
                print(' ', file=f)
            else:
                words = line.split()
                w1, w2, w3, w4 = words
                word, sim_cos = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)[0]
                print(' '.join([word, str(sim_cos)]), file=f)

'''   
19558it [1:21:33,  4.00it/s]
'''
'''
!paste questions-words.txt most_sim.txt > 64.txt
'''