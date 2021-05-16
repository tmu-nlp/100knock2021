# 17. １列目の文字列の異なり
# 1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはcut, sort, uniqコマンドを用いよ．


def sort_txt(pathfile):
   lines = [line.strip()for line in open(pathfile,'r',encoding='utf-8')]
   # print(set(lines))
   return len(set(lines))


if __name__ == '__main__':
    path = './col1.txt'
    print(sort_txt(path))