import pickle

def to_dict(line):
    line = line.rstrip()
    if line == "EOS":
        return None
    tmp = line.split("\t")
    pos = tmp[1].split(",")
    d = {"surface": tmp[0], "pos": pos[0], "pos1": pos[1], "base": pos[6]}
    return d

def mecab_to_list(text):
    l = []
    tmp = []
    for line in text:
        d = to_dict(line)
        if d is not None:
            tmp.append(d)
        elif tmp:
            l.append(tmp)
            tmp = []
    return l

table = {'\n', '\u3000'}

if __name__ == "__main__":
    text = open('data/neko.txt.mecab', 'r').readlines()
    text = [line for line in text if line not in table]
    ret = mecab_to_list(text)
    with open("./mecab.bin", "wb") as f:
        pickle.dump(ret, f)
