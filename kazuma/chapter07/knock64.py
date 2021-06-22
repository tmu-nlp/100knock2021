import urllib.request
from knock60 import load_vectors
from pprint import pprint
from tqdm import tqdm
    
def download_file():
    url = "http://download.tensorflow.org/data/questions-words.txt"
    urllib.request.urlretrieve(url, f"data/questions-words.txt")
    
def knock64():
    download_file()
    progress_bar = tqdm(total = 19558)
    word_vectors = load_vectors()
    with open("data/questions-words.txt", "r") as f1,\
         open("data/result_knock64.txt", "w") as f2:
        for line in f1:
            progress_bar.update(1)
            line = line.strip()
            words = line.split(" ")
            if len(words) != 4:
                f2.write(f"{line}\n")
                continue
            sim_word = word_vectors.most_similar(positive = [words[1], words[2]], negative = [words[0]], topn = 1)[0]
            f2.write(f"{line} {sim_word[0]} {sim_word[1]}\n")
    
if __name__ == "__main__":
    knock64()