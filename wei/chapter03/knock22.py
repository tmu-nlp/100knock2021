'''22. カテゴリ名の抽出
記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．
[ref]
https://docs.python.org/zh-cn/3/library/re.html#module-contents
[usage here]
(...) -> groups and capture,捕获分组从左括号开始计数，group 0代表整个表达式
         ((A)(B(C))) -> ((A)(B(C))) , (A), (B(C)), (C)
(?:...) -> 非捕获版本，匹配括号内的任何regex，但匹配的结果不能在 执行匹配后被捕获
\b ->  匹配空字符串，但只在单词开始或结尾的位置，通常定义为\w和\W 字符之间
\s ->  匹配任何Unicode空白字符，包括[\t \n \r \f \v]等
\S ->  匹配任何非空白字符，等价于[^\t \n \r \f \v]等
\w ->  匹配大小写字母、数字、下划线
\W ->  匹配非大小写字母、数字、下划线
.findall(pattern, string, flags = 0)
   -> 对str返回一个不重复的pattern匹配列表，从左到右扫描，匹配结果按找到的顺序返回；
      若pattern中存在 1或多个组，从左向右优先返回每个组匹配结果的列表；无括号时，返回整条语句匹配的结果'''



import re
from knock20 import read_gzip

if __name__ == '__main__':
    filepath = './data/jawiki-country.json.gz'
    text = read_gzip(filepath, 'イギリス')              # str
    # print(text)

    pattern = r'\[\[Category:(.*?)(?:\|.*\]\]|\]\])'   # [[Category:(イギリス)(|*]] / ]])
    categories = re.findall(pattern, text)             # list
    for category in categories:
        print(category)