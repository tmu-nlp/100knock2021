from os import write
from knock41 import sentence_set

with open('./ans47.txt','w')as f:
    for sentence in sentence_set:
        for chunk in sentence.chunks:
            for morph in chunk.morphs:
                if morph.pos=='動詞':
                    for i,src in enumerate(chunk.srcs):
                        if len(sentence.chunks[src].morphs)==2 and sentence.chunks[src].morphs[0].pos1=='サ変接続' and sentence.chunks[src].morphs[1].surface=='を':
                            #pred=[サ変接続名詞]＋［を］＋「動詞」
                            pred=''.join([sentence.chunks[src].morphs[0].surface, sentence.chunks[src].morphs[1].surface, morph.base])
                            case=[]
                            modi_chunks=[]
                            for src_rest in chunk.srcs[:i]+chunk.srcs[i+1:]:
                                #遍历所有src係元chunk的morph，寻找所有pos为助词的morph的surface，加到case中
                                tmpcase=[morph.surface for morph in sentence.chunks[src_rest].morphs if morph.pos=='助詞']
                                if len(tmpcase)>0:
                                    case+=tmpcase
                                    modi_chunks.append(''.join(morph.surface for morph in sentence.chunks[src_rest].morphs if morph.pos!='記号'))
                            if len(case)>0:
                                case=sorted(list(set(case)))
                                line = '{}\t{}\t{}'.format(pred, ' '.join(case), ' '.join(modi_chunks))
                                f.write(line+'\n')
                            break
            