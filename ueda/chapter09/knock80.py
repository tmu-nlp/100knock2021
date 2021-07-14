#knock80.py
import gensim
import gzip
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import Counter
import torch
from torch import nn

def create_wordids(file_name):
    counter = Counter()
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split('\t')[1]
            for word in line.split():
                counter[word] += 1
    wordids = {}
    ids=0
    for word, count in counter.most_common():
        if count < 2: continue
        ids += 1
        wordids[word] = ids
    return wordids

def sentence2ids(wordids, line):
    line = line.split(" ")
    for i in range(len(line)):
        if line[i] in wordids:
            line[i] = wordids[line[i]]
        else:
            line[i] = 0
    return line
            
wordids = create_wordids(r'C:\Git\train.txt')
print(sentence2ids(wordids, 'Independence Day Pump Prices May Hit 6-Year High on Iraq'))