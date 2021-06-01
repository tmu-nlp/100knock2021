from os import write
from knock41 import sentence_set
with open('./ans46.txt','w')as f:
    for sentence in sentence_set:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos=='動詞':
                    case=[]
                    modi_chunks=[]
                    for src in chunk.srcs:
                        #遍历所有src係元chunk的morph，寻找所有pos为助词的morph的surface，加到case中
                        tmpcase=[morph.surface for morph in sentence.chunks[src].morphs if morph.pos=='助詞']
                        if len(tmpcase)>0:
                            case+=tmpcase
                            modi_chunks.append(''.join(morph.surface for morph in sentence.chunks[src].morphs if morph.pos!='記号'))
                    if len(case)>0:
                        case=sorted(list(set(case)))
                        line = '{}\t{}\t{}'.format(morph.base, ' '.join(case), ' '.join(modi_chunks))
                        f.write(line+'\n')
                    break
            