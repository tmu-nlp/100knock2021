class Morph:
    #インスタンスの初期化
    def __init__(self,morph):
        surface,attr=morph.split('\t')
        attr=attr.split(',')
        self.surface=surface
        self.base=attr[6]
        self.pos=attr[0]
        self.pos1=attr[1]

file_name="./ai.ja.txt.parse"

sentence=[]
morph=[]
with open(file_name,'r') as f:
    for line in f:
        if line[0]=='*':
            continue#＊からの行をスキップする
        elif line !='EOS\n':
            morph.append(Morph(line))
        else:
            sentence.append(morph)#１行の文の解析結果を保存する
            morph=[]

for s in sentence[:3]:
    for w in s:
        print(w.surface+' '+w.base+' '+w.pos+' '+w.pos1+'\n')