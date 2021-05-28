#“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”
#という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

Input = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

text = Input.replace(',', '').replace('.', '')
words = text.split(" ")

ans = [len(w) for w in words]
print(ans)