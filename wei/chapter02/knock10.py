# 10. 行数のカウント
# 行数をカウントせよ．確認にはwcコマンドを用いよ.

import os


def count_line(path):

    line_num = 0
    for i, line in enumerate(open(path,"r",encoding='utf-8')):
        line_num += 1
        # print(line_num,line)        # print the number of lines and its contents
    return line_num


if __name__ == '__main__':
    file = './popular-names.txt'
    # print(os.getcwd())             # get file path
    print('the number of lines is:', count_line(file))

