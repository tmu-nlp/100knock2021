'''
!mkdir 94

#beam size = 1
!fairseq-interactive \
  --path checkpoints/kftt.ja-en/checkpoint_best.pt \
  --beam 1 data-bin/kftt.ja-en/ \
  < kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94/beam1.out
!fairseq-score --sys 94/beam1.out --ref kftt-data-1.0/data/tok/kyoto-dev.en
#出力
["Namespace(ignore_case=False, order=4, ref='kftt-data-1.0/data/tok/kyoto-dev.en', sacrebleu=False, sentence_bleu=False, sys='94/beam1.out')",
 'BLEU4 = 7.49, 31.1/10.8/4.5/2.1 (BP=1.000, ratio=1.331, syslen=32367, reflen=24309)']

#beam size = 5
!fairseq-interactive \
  --path checkpoints/kftt.ja-en/checkpoint_best.pt \
  --beam 5 data-bin/kftt.ja-en/ \
  < kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94/beam5.out
!fairseq-score --sys 94/beam5.out --ref kftt-data-1.0/data/tok/kyoto-dev.en
#出力
["Namespace(ignore_case=False, order=4, ref='kftt-data-1.0/data/tok/kyoto-dev.en', sacrebleu=False, sentence_bleu=False, sys='94/beam5.out')",
 'BLEU4 = 9.70, 36.5/13.4/6.0/3.0 (BP=1.000, ratio=1.137, syslen=27640, reflen=24309)']

#beam size = 10
!fairseq-interactive \
  --path checkpoints/kftt.ja-en/checkpoint_best.pt \
  --beam 10 data-bin/kftt.ja-en/ \
  < kftt-data-1.0/data/tok/kyoto-dev.ja | grep '^H' | cut -f3 > 94/beam10.out
!fairseq-score --sys 94/beam10.out --ref kftt-data-1.0/data/tok/kyoto-dev.en
#出力
["Namespace(ignore_case=False, order=4, ref='kftt-data-1.0/data/tok/kyoto-dev.en', sacrebleu=False, sentence_bleu=False, sys='94/beam10.out')",
 'BLEU4 = 9.63, 35.8/13.3/6.0/3.0 (BP=1.000, ratio=1.149, syslen=27943, reflen=24309)']

''' 