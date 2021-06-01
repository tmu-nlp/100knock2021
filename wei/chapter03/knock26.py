""'''26. 強調マークアップの除去
25の処理時に，テンプレートの値からMediaWikiの強調マークアップ
（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ

[ref]
- re.sub(pattern, repl, string, count=0, flags=0)
  - 返回通过使用repl 替换在 string 最左边非重叠出现的 pattern而获得的字符串。
    若pattern没找到，则不加改变地返回string。
  
  
  '''

