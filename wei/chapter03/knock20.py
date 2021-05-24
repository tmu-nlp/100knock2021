'''20. JSONデータの読み込み
Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
問題21-29では，ここで抽出した記事本文に対して実行せよ．'''

import json
import gzip


def read_gzip(Filename, title):
    with gzip.open(Filename) as f:

        for line in f:
            obj = json.loads(line)                   # json.read()读取多行json文件时，单行读取文件，解码为字典对象，字典长度为2

            if obj['title'] == title:
                return obj['text']         # 返回该dict中 key为text的value，type为string。


if __name__ == '__main__':
    filepath = './data/jawiki-country.json.gz'
    text = read_gzip(filepath, 'イギリス')        # str
    UK_info = './data/UK_info.json'
    with open(UK_info,'w+',encoding='utf-8') as outFile:
        outFile.write(text)

    print(text)