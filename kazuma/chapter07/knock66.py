from knock60 import load_vectors
from pprint import pprint
import urllib.request
import zipfile
import os
from scipy.stats import spearmanr


def download_file():
    url = "http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip"
    urllib.request.urlretrieve(url, "data/wordsim-353.zip")

def unpack_zip():
    file_name = "wordsim-353.zip"
    with zipfile.ZipFile(f"data/{file_name}") as existing_zip:
        existing_zip.extractall('data')
    # os.remove(f"data/{file_name}")

def knock66():
    # download_file()
    # unpack_zip()
    word_vectors = load_vectors()
    human_scores = []
    w2v_scores = []
    with open("data/set1.tab", "r") as f1:
        for i, line in enumerate(f1):
            if i == 0:continue
            word1, word2, human_score, *_ = line.strip().split("\t")
            w2v_score = word_vectors.similarity(word1, word2)
            human_scores.append(human_score)
            w2v_scores.append(w2v_score)
    print(spearmanr(human_scores, w2v_scores)[0])

if __name__ == "__main__":
    knock66()

