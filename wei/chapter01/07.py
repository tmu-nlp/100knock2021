# 07. テンプレートによる文生成
# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y="気温", z=22.4として，実行結果を確認せよ．

# .formatによる文字列のフォーマット

def current_tempre(x, y, z):
    return '{}時の{}は{}'.format(x, y, z)


if __name__ == '__main__':
    print
    current_tempre(12, '気温', 22.4)