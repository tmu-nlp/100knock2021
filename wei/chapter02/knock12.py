# 12. 1列目をcol1.txtに，2列目をcol2.txtに保存
# 各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．確認にはcutコマンドを用いよ．


def col_list(path):
    result = []

    for line in open(path,'r',encoding='utf-8').readlines():
        newline = line.strip().split()
        result.append(newline)
        print(result)

    return result




if __name__ == '__main__':
    file = './test.txt'
    list_result = col_list(file)
    with open('./col1.txt', 'w+', encoding= 'utf-8') as col1_out, open('./col2.txt', 'w+', encoding= 'utf-8') as col2_out:
        for i in list_result:
            col1_out.write(i[0]+'\n')
            col2_out.write(i[1]+'\n')
