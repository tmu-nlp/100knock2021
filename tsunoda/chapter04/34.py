filename = 'data/neko.txt.mecab'

sentences = []

morphs = []
with open(filename, mode='r', encoding="utf-8") as f:
  for line in f:  # 1行ずつ読込
    if line != 'EOS\n':  # 文末以外：形態素解析情報を辞書型に格納して形態素リストに追加
      fields = line.split('\t')
      if len(fields) != 2 or fields[0] == '':  # 文頭以外の空白と改行文字はスキップ
        continue
      else:
        attr =  fields[1].split(',')
        morph = {'surface': fields[0], 'base': attr[6], 'pos': attr[0], 'pos1': attr[1]}
        morphs.append(morph)
    else:  # 文末：形態素リストを文リストに追加
      sentences.append(morphs)
      morphs = []

ans = set()
for sentence in sentences:
  nouns = ''
  num = 0
  for morph in sentence:
    if morph['pos'] == '名詞': # 最初の形態素から順に、名詞であればnoun
      nouns = ''.join([nouns, morph['surface']])
      num += 1
    elif num >= 2: # 名詞以外、かつここまでの連結数が2以上の場合は出力し、nounsとnumを初期化
      ans.add(nouns)
      nouns = ''
      num = 0
    else:  # それ以外の場合、nounsとnumを初期化
      nouns = ''
      num = 0
  if num >= 2:
    ans.add(nouns)

#　確認
print(f'連投名詞の種類: {len(ans)}\n')
print('---サンプル---')
for n in list(ans)[:10]:
  print(n)
  