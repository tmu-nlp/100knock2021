# 95. サブワード化
# トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．

import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')

sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')

for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.ja', 'train.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test.sub.ja'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)

#▁ 雪 舟 ( せ っ しゅう 、 14 20 年 ( 応永 27 年 )- 150 6 年 ( 永正 3 年 ) ) は 号 で 、 15 世紀後半 室町時代に 活躍した 水墨画 家 ・ 禅僧 で 、 画 聖 とも 称え られる 。
#▁日本の 水墨画 を 一 変 させた 。
#▁諱は 「 等 楊 ( とう よう ) 」 、 もしくは 「 拙 宗 ( せ っ しゅう ) 」 と号した 。
#▁ 備中国 に 生まれ 、 京都 ・ 相国寺 に入って から 周防国 に移る 。
#▁その後 遣 明 使 に 随行 して 中国 ( 明 ) に渡って 中国の 水墨画 を学んだ 。
#▁ 作品 は 数多く 、 中国 風の 山 水 画 だけでなく 人物 画 や 花鳥 画 も よく した 。
#▁大 胆 な 構図 と 力 強い 筆 線 は非常に 個 性 的な 画 風 を作り 出している 。
#▁ 現存する 作品 のうち 6 点 が 国宝 に指定され ており 、 日本の 画家 のなかで も 別 格 の 評価 を受けている といえる 。
#▁このため 、 花鳥 図屏風 などに 「 伝 雪 舟 筆 」 される 作品 は 大変 多い 。
#▁ 真 筆 である か 専門 家 の間で も 意見 の 分かれ るもの も 多 々 ある 。


subword-nmt learn-bpe -s 16000 < kftt-data-1.0/data/orig/kyoto-train.en > kyoto_en.codes
subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-train.en > train.sub.en
subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-dev.en > dev.sub.en
subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-test.en > test.sub.en


#K@@ n@@ own as Ses@@ shu (14@@ 20 - 150@@ 6@@ ), he was an ink painter and Zen monk active in the Muromachi period in the latter half of the 15th century, and was called a master pain@@ ter.
#He revol@@ ut@@ ion@@ ized the Japanese ink paint@@ ing.
#He was given the posthumous name "@@ Toyo@@ " or "S@@ es@@ shu (@@ 拙@@ 宗@@ )."
#Born in Bicchu Province, he moved to Suo Province after entering S@@ Shokoku-ji Temple in Kyoto.
#Later he accompanied a mission to Ming Dynasty China and learned Chinese ink paint@@ ing.
#His works were man@@ y, including not only Chinese-style landscape paintings, but also portraits and pictures of flowers and bird@@ s.
#His b@@ old compos@@ itions and strong brush st@@ rok@@ es const@@ ituted an extremely distinctive style.
#6 of his ext@@ ant works are designated national treasu@@ res. In@@ de@@ ed, he is considered to be extraordinary among Japanese pain@@ ters.
#For this reason, there are a great many art@@ works that are attributed to him, such as folding scre@@ ens with pictures of flowers and that birds are painted on them.
#There are many works that even experts cannot ag@@ ree if they are really his work or not.

#前処理
fairseq-preprocess -s ja -t en \
    --trainpref train.sub \
    --validpref dev.sub \
    --destdir data95  \
    --workers 20
#訓練をして
fairseq-train data95 \
    --fp16 \
    --save-dir save95 \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --update-freq 1 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 95.log
#生成して
fairseq-interactive --path save95/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 95.out
#トークナイズをSpaCyにあわせて
def spacy_tokenize(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = ' '.join([doc.text for doc in nlp(x)])
            print(x, file=g)
spacy_tokenize('95.out', '95.out.spacy')
#スコアを測定します
fairseq-score --sys 95.out.spacy --ref test.spacy.en

Namespace(ignore_case=False, order=4, ref='test.spacy.en', sacrebleu=False, sentence_bleu=False, sys='95.out.spacy')
BLEU4 = 20.36, 51.3/25.2/14.7/9.0 (BP=1.000, ratio=1.030, syslen=28463, reflen=27625)

for N in `seq 1 10` ; do
    fairseq-interactive --path save95/checkpoint10.pt --beam $N data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 95.$N.out
done

for i in range(1, 11):
    spacy_tokenize(f'95.{i}.out', f'95.{i}.out.spacy')

for N in `seq 1 10` ; do
    fairseq-score --sys 95.$N.out.spacy --ref test.spacy.en > 95.$N.score
done

xs = range(1, 11)
ys = [read_score(f'95.{x}.score') for x in xs]
plt.plot(xs, ys)
plt.show()