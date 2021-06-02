from knock41 import sentences

for sentence in sentences:
    verb = ''; v_srcs = []
    particles = []
    arguments = []
    for i, chunk in enumerate(sentence):

        #動詞決定
        for morph in chunk.morphs:
            if morph.pos == '動詞':
                verb = morph.base
                v_srcs += chunk.srcs
                
            if verb == '':
                continue
                
            #格パターン、項の抽出
            for src in v_srcs:
                particle = []
                argument = set()
                for i, morph in enumerate(sentence[src].morphs):
                    if i == 0:
                        continue

                    if sentence[src].morphs[i-1].pos1 == 'サ変接続' and morph.surface == 'を':
                        verb = sentence[src].morphs[i-1].surface + morph.surface + verb
                        #「サ変接続+を」がかかる時の文節
                        for morph in sentence[src].morphs:
                            if morph.pos == '助詞':
                                particle.append(morph.surface)
                                argument.add(sentence[src].phrase)

                particles.extend(particle)
                arguments.extend(list(argument))

            particles = '\t'.join(particles)
            arguments = '\t'.join(arguments)

            if len(particles) > 0:
                print(f'{verb}\t{particles}\t{arguments}')

            verb = ''; v_srcs = []
            particles = []
            arguments = []