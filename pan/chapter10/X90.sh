# 90. データの準備
# 機械翻訳のデータセットをダウンロードせよ．訓練データ，開発データ，評価データを整形し，必要に応じてトークン化などの前処理を行うこと．
# ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ．

# KFTTデータをダウンロードして解凍します
tar zxvf kftt-data-1.0.tar.gz
# GiNZAで日本語側のデータをトークナイズします
cat kftt-data-1.0/data/orig/kyoto-train.ja | sed 's/\s+/ /g' | ginzame > train.ginza.ja
cat kftt-data-1.0/data/orig/kyoto-dev.ja | sed 's/\s+/ /g' | ginzame > dev.ginza.ja
cat kftt-data-1.0/data/orig/kyoto-test.ja | sed 's/\s+/ /g' | ginzame > test.ginza.ja

# 日本語
for src, dst in [
    ('train.ginza.ja', 'train.spacy.ja'),
    ('dev.ginza.ja', 'dev.spacy.ja'),
    ('test.ginza.ja', 'test.spacy.ja'),
]:
    with open(src) as f:
        lst = []
        tmp = []
        for x in f:
            x = x.strip()
            if x == 'EOS':
                lst.append(' '.join(tmp))
                tmp = []
            elif x != '':
                tmp.append(x.split('\t')[0])
    with open(dst, 'w') as f:
        for line in lst:
            print(line, file=f)

#雪舟 （ せっ しゅう 、 1420 年 （ 応永 27 年 ） - 1506 年 （ 永正 3 年 ） ） は 号 で 、 15 世紀 後半 室町 時代 に 活躍 し た 水墨 画家 ・ 禅僧 で 、 画聖 と も 称え られる 。
#日本 の 水墨画 を 一変 さ せ た 。
#諱 は 「 等楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっ しゅう ） 」 と 号し た 。
#備中 国 に 生まれ 、 京都 ・ 相国 寺 に 入っ て から 周防 国 に 移る 。
#その 後 遣明 使 に 随行 し て 中国 （ 明 ） に 渡っ て 中国 の 水墨画 を 学ん だ 。
#作品 は 数多く 、 中国 風 の 山水画 だけ で なく 人物画 や 花鳥画 も よく し た 。
#大胆 な 構図 と 力強い 筆線 は 非常 に 個性的 な 画風 を 作り出し て いる 。
#現存 する 作品 の うち 6 点 が 国宝 に 指定 さ れ て おり 、 日本 の 画家 の なか で も 別格 の 評価 を 受け て いる と いえる 。
#この ため 、 花鳥 図 屏風 など に 「 伝 雪舟 筆 」 さ れる 作品 は 大変 多い 。
#真筆 で ある か 専門家 の 間 で も 意見 の 分かれる もの も 多々 ある 。

#英語
import re
import spacy

nlp = spacy.load('en')
for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.en', 'train.spacy.en'),
    ('kftt-data-1.0/data/orig/kyoto-dev.en', 'dev.spacy.en'),
    ('kftt-data-1.0/data/orig/kyoto-test.en', 'test.spacy.en'),]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = nlp.make_doc(x)
            x = ' '.join([doc.text for doc in x])
            print(x, file=g)

#Known as Sesshu ( 1420 - 1506 ) , he was an ink painter and Zen monk active in the Muromachi period in the latter half of the 15th century , and was called a master painter .
#He revolutionized the Japanese ink painting .
#He was given the posthumous name " Toyo " or " Sesshu ( 拙宗 ) . "
#Born in Bicchu Province , he moved to Suo Province after entering SShokoku - ji Temple in Kyoto .
#Later he accompanied a mission to Ming Dynasty China and learned Chinese ink painting .
#His works were many , including not only Chinese - style landscape paintings , but also portraits and pictures of flowers and birds .
#His bold compositions and strong brush strokes constituted an extremely distinctive style .
#6 of his extant works are designated national treasures . Indeed , he is considered to be extraordinary among Japanese painters .
#For this reason , there are a great many artworks that are attributed to him , such as folding screens with pictures of flowers and that birds are painted on them .
#There are many works that even experts can not agree if they are really his work or not .