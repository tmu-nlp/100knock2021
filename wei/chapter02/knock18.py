''' 18. 各行を3コラム目の数値の降順にソー
各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．
確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい） '''

import numpy as np
import pprint


def sort_by_col(pathfile, n):
    lines = [tuple(line.strip().split()) for line in open(pathfile, encoding='utf-8')]
    # print(lines)
    t = np.dtype([('KEN', str, 20), ('SHI', str, 20), ('temp', float), ('date', str, 20)])
    a = np.array(lines, dtype=t)
    b = a.tolist()
    # print(b)
    result = sorted(b, key=lambda x: (x[n], x[0]), reverse=True)
    return result


if __name__ == '__main__':
    path = './popular-names.txt'
    sorted_names_list = sort_by_col(path, 2)
    sorted_list = np.array(sorted_names_list)
    # print(sorted_list)
    with open('./knock18.txt', 'w', encoding='utf-8') as f:
        for i in sorted_list:
            n_line = ' '.join(list(i))
            f.write(n_line + '\n')