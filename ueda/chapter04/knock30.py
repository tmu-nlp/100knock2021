def load_mecab():
    with open(r'c:\Git\neko.txt.mecab', encoding="utf-8") as f:
        morpheme_list= []
        for line in f:
            col = line.split('\t')
            if col[0]=='EOS\n' and len(morpheme_list) != 0:
                yield morpheme_list
                morpheme_list = []
            elif col[0]!='EOS\n':
                other = col[1].split(',')
                morpheme = {'surface': col[0], 'base':other[6], 'pos':other[0], 'pos1':other[1]}
                morpheme_list.append(morpheme)

if __name__ == "__main__":
    for line in load_mecab():
        print(line)