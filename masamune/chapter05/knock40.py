class Morph:
    def __init__(self, morph):
        self.surface = morph[0]
        self.base = morph[7]
        self.pos = morph[1]
        self.pos1 = morph[2]

morphs = [] #形態素
sentences = [] #文
with open('ai.ja.txt.parsed') as file:
    for line in file:
        if line[0] == '*' or line == '\n':
            continue

        #EOSの時に形態素をsentenceに追加
        elif line == 'EOS\n':
            if len(morphs) > 0:
                sentences.append(morphs)
                morphs = []

        #形態素をmorphsに追加
        else:
            line = line.replace('\n', '').split('\t')
            morph = [line[0]]
            morph.extend(line[1].split(',')) 
            morphs.append(Morph(morph))

if __name__ == '__main__':
    for sentence in sentences:
        for morph in sentence:
            print(morph.__dict__)