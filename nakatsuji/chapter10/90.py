'''
英語
cat ktff-data-1.0/data/orig/kyoto-train.en | moses-tokenizer en > keff_data98/train.tok.en
cat ktff-data-1.0/data/orig/kyoto-dev.en | moses-tokenizer en > keff_data98/dev.tok.en
cat ktff-data-1.0/data/orig/kyoto-test.en | moses-tokenizer en > keff_data98/test.tok.en

日本語
python tokja.py
'''