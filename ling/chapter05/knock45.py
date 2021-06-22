from knock41 import sentence_set
with open('./ans45.txt','w')as f:
    for sentence in sentence_set:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos=='動詞':
                    case=[]
                    for src in chunk.srcs:
                        #遍历所有src中的chunk的morph，寻找所有pos为助词的morph的surface，加到case中
                        case+=[morph.surface for morph in sentence.chunks[src].morphs if morph.pos=='助詞']
                    if len(case)>0:
                        case=sorted(list(set(case)))
                        line=morph.base+'\t'+' '.join(case)
                        f.write(line+'\n')
                    break
            