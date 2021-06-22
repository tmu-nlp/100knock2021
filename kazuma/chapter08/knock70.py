import sys
sys.path.append('../')
from chapter07.knock60 import load_vectors
import numpy as np

def knock70():
    with open("../chapter06/data/train.txt", "r") as f1,\
        open("../chapter06/data/valid.txt", "r") as f2,\
        open("../chapter06/data/test.txt", "r") as f3,\
        open("data/train_fea.txt", "w") as f4,\
        open("data/valid_fea.txt", "w") as f5,\
        open("data/test_fea.txt", "w") as f6,\
        open("data/train_label.txt", "w") as f7,\
        open("data/valid_label.txt", "w") as f8,\
        open("data/test_label.txt", "w") as f9:
        word_vectors = load_vectors()
        dict1 = {'b':0, 't':1, 'e':2, 'm':3}
        for line in f1:
            words, label = line.strip().split("\t")
            words = words.split(" ")
            x_i = np.zeros(300)
            cnt = 0
            for word in words: # 内包表記とsumとlen使ってもっとpythonらしくかけそうな気もする。今度。
                if word in word_vectors:
                    cnt += 1
                    x_i += word_vectors[word]
            if cnt:
                x_i /= cnt
            x_i = " ".join([str(ele) for ele in x_i])
            f4.write(f"{x_i}\n")
            f7.write(f"{dict1[label]}\n")
        
        for line in f1:
            words, label = line.strip().split("\t")
            words = words.split(" ")
            x_i = np.zeros(300)
            cnt = 0
            for word in words: # 内包表記とsumとlen使ってもっとpythonらしくかけそうな気もする。今度。
                if word in word_vectors:
                    cnt += 1
                    x_i += word_vectors[word]
            if cnt:
                x_i /= cnt
            x_i = " ".join([str(ele) for ele in x_i])
            f4.write(f"{x_i}\n")
            f7.write(f"{dict1[label]}\n")
        
        for line in f2:
            words, label = line.strip().split("\t")
            words = words.split(" ")
            x_i = np.zeros(300)
            cnt = 0
            for word in words: # 内包表記とsumとlen使ってもっとpythonらしくかけそうな気もする。今度。
                if word in word_vectors:
                    cnt += 1
                    x_i += word_vectors[word]
            if cnt:
                x_i /= cnt
            x_i = " ".join([str(ele) for ele in x_i])
            f5.write(f"{x_i}\n")
            f8.write(f"{dict1[label]}\n")

        for line in f3:
            words, label = line.strip().split("\t")
            words = words.split(" ")
            x_i = np.zeros(300)
            cnt = 0
            for word in words: # 内包表記とsumとlen使ってもっとpythonらしくかけそうな気もする。今度。
                if word in word_vectors:
                    cnt += 1
                    x_i += word_vectors[word]
            if cnt:
                x_i /= cnt
            x_i = " ".join([str(ele) for ele in x_i])
            f6.write(f"{x_i}\n")
            f9.write(f"{dict1[label]}\n")
        
if __name__  == "__main__":
    knock70()