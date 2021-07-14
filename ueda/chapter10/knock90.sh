#knock90
#kftt-data-1.0.tar.gzの中に良さそうなデータがあるのでそれを使う
!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref data/kyoto-train.cln --validpref data/kyoto-dev --testpref data/kyoto-test \
    --destdir data/kyoto --workers 10