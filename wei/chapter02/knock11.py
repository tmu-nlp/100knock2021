# 11. タブをスペースに置換
# タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．


def tab2space(path):
    with open(path, 'r', encoding='utf-8') as file:         # 打开文本对象,操作结束后自动关闭。
        data = file.read()
        new_f = data.replace('\t', ' ')                     # .read()读取文本内容为字符串，可对字符串操作，但不被保存; .replace()替换字符串中指定内容

    with open(path, 'w+', encoding='utf-8') as new_file:
        new_file.write(new_f)                                # 打开文件，写入内容，覆盖原本内容
    return new_file


if __name__ == '__main__':
    file = './popular-names.txt'
    tab2space(file)
    nfile = open(file, 'r').read()
    # print(nfile)

