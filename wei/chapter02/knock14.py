# 14. 先頭からN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ．


import itertools

def read_N(path, N):
    with open(path, 'r',encoding='utf-8') as f:
        for line in itertools.islice(f, 0, N):
            print(line.strip())



if __name__ == '__main__':
    file = './popular-names.txt'
    read_N(file,6)
   


