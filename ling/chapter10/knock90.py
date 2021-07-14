def show_top_10_lines(path):
    with open(path,'r') as f:
        data=f.readlines()
        for line in data[:10]:
            print(line)
train_tok_path='/Users/lingzhidong/Documents/GitHub/100knock2021/ling/chapter10/kftt-data-1.0/data/tok/kyoto-train.cln.ja'
test_tok_path='/Users/lingzhidong/Documents/GitHub/100knock2021/ling/chapter10/kftt-data-1.0/data/tok/kyoto-test.ja'
valid_tok_path='/Users/lingzhidong/Documents/GitHub/100knock2021/ling/chapter10/kftt-data-1.0/data/tok/kyoto-dev.ja'

show_top_10_lines(train_tok_path)
show_top_10_lines(test_tok_path)
show_top_10_lines(valid_tok_path)