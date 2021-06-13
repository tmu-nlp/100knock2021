'''
FILENAME #1: newsCorpora.csv (102.297.000 bytes)
DESCRIPTION: News pages
FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP

where:
ID		Numeric ID
TITLE		News title 
URL		Url
PUBLISHER	Publisher name
CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)
STORY		Alphanumeric ID of the cluster that includes news about the same story
HOSTNAME	Url hostname
TIMESTAMP 	Approximate time the news was published, as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970


FILENAME #2: 2pageSessions.csv (3.049.986 bytes)
DESCRIPTION: 2-page sessions
FORMAT: STORY \t HOSTNAME \t CATEGORY \t URL

where:
STORY		Alphanumeric ID of the cluster that includes news about the same story
HOSTNAME	Url hostname
CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)
URL		Two space-delimited urls representing a browsing session
'''
import zipfile
import urllib.request
import re
import os
import random
from pprint import pprint
from collections import defaultdict
import pickle

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
    file_name = re.search(r".*/(.*\.zip)", url).group(1)
    urllib.request.urlretrieve(url, f"data/{file_name}")
    return file_name

def unpack_zip(file_name):
    with zipfile.ZipFile(f"data/{file_name}") as existing_zip:
        existing_zip.extractall('data')
    os.remove(f"data/{file_name}")
    return file_name

def get_target_contents():
    file_name = "data/newsCorpora.csv"
    target_publisher = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
    target_contents = []
    with open(file_name, "r") as f:
        for line in f:
            if line.split("\t")[3] in target_publisher:
                target_contents.append(line)
    return target_contents 

def shuffle_contents(contents):
    random.shuffle(contents)
    return contents

def split_doc_and_extract_catecory_and_title(contents):
    with open("data/train.txt", "w") as f1, \
         open("data/valid.txt", "w") as f2, \
         open("data/test.txt", "w") as f3:
        cnt = 0
        for i in range(int(len(contents)*0.8)):
            cnt += 1
            content = contents[i].split('\t')
            f1.write(f"{content[1]}\t{content[4]}\n")
        print(cnt)
        cnt = 0
        for i in range(int(len(contents)*0.8), int(len(contents)*0.9)):
            cnt += 1
            content = contents[i].split('\t')
            f2.write(f"{content[1]}\t{content[4]}\n")
        print(cnt)
        cnt = 0
        for i in range(int(len(contents)*0.9), len(contents)):
            cnt += 1
            content = contents[i].split('\t')
            f3.write(f"{content[1]}\t{content[4]}\n")
        print(cnt)

def check_category():
    with open("data/train.txt", "r") as f1, \
         open("data/valid.txt", "r") as f2, \
         open("data/test.txt", "r") as f3:
        d1, d2, d3= defaultdict(lambda:[0]), defaultdict(lambda:[0]), defaultdict(lambda:[0])
        l1, l2, l3 = 0,0,0
        for i, line in enumerate(f1):
            d1[line.split("\t")[0]][0] += 1
            l1 = i
        for i, line in enumerate(f2):
            d2[line.split("\t")[0]][0] += 1
            l2 = i
        for i, line in enumerate(f3):
            d3[line.split("\t")[0]][0] += 1
            l3 = i
        for k, v in d1.items():
            d1[k] += [round(v[0]/l1, 2)]
        for k, v in d2.items():
            d2[k] += [round(v[0]/l2, 2)]
        for k, v in d3.items():
            d3[k] += [round(v[0]/l3, 2)]
        # pprint(d1)
        # pprint(d2)
        # pprint(d3)


if __name__ == "__main__":
    # # step1
    # unpack_zip(download_data())

    # step2
    target_contents = get_target_contents()

    # step3
    shuffle_contents(target_contents)

    # step4
    split_doc_and_extract_catecory_and_title(target_contents)

    check_category()