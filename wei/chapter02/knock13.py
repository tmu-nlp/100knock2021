# 13. col1.txtとcol2.txtをマージ
# 12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．確認にはpasteコマンドを用いよ．


def new_txt(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f1, open(path2,'r', encoding= 'utf-8') as f2:
        f1_list = f1.readlines()
        f2_list = f2.readlines()
    return [f1_list, f2_list]                         #返回[[ ],[ ]]


if __name__ == '__main__':
    file1,file2 = './col1.txt','./col2.txt'
    n_txt = new_txt(file1,file2)

    with open('./knock13.txt','w+',encoding='utf-8') as f_out:
        for i1,i2 in zip(n_txt[0],n_txt[1]):                     # zip(arg1,arg2)分别取遍list arg1和list arg2中每个元素
            n_line = i1.strip() + '\t' + i2.strip() + '\n'
            # print(n_line)
            f_out.write(n_line)

