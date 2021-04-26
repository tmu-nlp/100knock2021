
a = 'Hi He Lied Because Boron Could Not Oxidize Fluorune. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
b = a.split()
one = [1, 5, 6, 7, 8, 9, 15, 16, 19]  # 1文字を取り出す単語の番号リスト
ans = {}
for i, word in enumerate(b):
  if i + 1 in one:
    ans[word[:1]] = i + 1  # リストにあれば1文字を取得
  else:
    ans[word[:2]] = i + 1  # なければ2文字を取得

print(ans)