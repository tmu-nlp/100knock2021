'''19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．確認にはcut, uniq, sortコマンドを用いよ．'''


def col1_count(pathfile):
    lines = [line.strip().split() for line in open(pathfile, encoding='utf-8')]
    col1_list = []
    for i in lines:
        col1_list.append(i[0])
    count_col1 = {}
    for i in set(col1_list):
        # 以集合中不重复元素i为字典的键，以原列表中该元素i出现的次数为值
        count_col1[i] = col1_list.count(i)
    # 按照出现频次由高到低，以元组形式显示城市和出现次数，返回列表。
    return sorted(count_col1.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    path = './knock18.txt'
    print(col1_count(path))