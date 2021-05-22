import MeCab
file_path = 'data/neko.txt'
#空のリストの用意
m_list = []
#テキストデータの読み込み
with open(file_path, encoding="utf-8") as f:
  text_list = f.read()
  #改行で切り分けて各行ごとに形態素解析を行う
  for i in text_list.split('\n'):
    mecab = MeCab.Tagger('mecab-ipadic-neologd').parse(i)
    #用意したm_listに格納
    m_list.append(mecab)
  print(m_list[2])

path_w = 'data/neko.txt.mecab'

with open(path_w, encoding="utf-8", mode='w') as f:
	f.writelines(m_list)