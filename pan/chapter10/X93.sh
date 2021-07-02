# 93. BLEUスコアの計測
# 91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．

fairseq-score --sys 92.out --ref test.spacy.en