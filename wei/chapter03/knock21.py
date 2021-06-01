'''21. カテゴリ名を含む行を抽出
記事中でカテゴリ名を宣言している行を抽出せよ．
[ref]
https://docs.python.org/zh-cn/3/library/re.html#module-contents

[regex used here]
. -> 匹配除\n外的任意一个字符
$ -> 匹配字符串尾 或 字符串尾的换行符的前一个字符
* -> 匹配之前的regex 0到任意次重复   greedy match
+ -> 匹配之前的regex 1到任意次重复   greedy match
? -> 匹配之前的regex 0到1次         greedy match
*?,+?,??  -> 变为非贪婪匹配         e.g. 使用<.*?>匹配字符串'<a>b<c>，结果为<a>
[] -> 表 字符集合，在集合中，特殊字符将失去特殊含义。
                         如[(+*)]将匹配字符 '(','+','*',or ')'
[^ad] -> 取反匹配，将匹配集合中任意一个字符，匹配除'ad'外的任意一个字符
(...) -> 匹配括号内任意regex，标识出组合start和end。完成匹配后，组合内容被获取'''



import re
from knock20 import read_gzip

if __name__ == '__main__':
    filepath = './data/jawiki-country.json.gz'
    text = read_gzip(filepath, 'イギリス')

    pattern = r'\[\[Category:.*?\]\]'
    categories = re.findall(pattern, text)   # is a list
    print('\n'.join(categories))             # is a str, 将列表中的元素按行显示

