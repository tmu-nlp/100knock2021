from knock41 import sentences

with open('verb_particle.txt', 'w') as o_file:
    for sentence in sentences:
        verb = ''; v_srcs = []
        particles = []
        for chunk in sentence:

            #動詞決定
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    verb = morph.base
                    v_srcs += chunk.srcs

                if verb == '':
                    continue

                #格パターン抽出
                for src in v_srcs:
                    particle = []
                    for morph in sentence[src].morphs:
                        if morph.pos == '助詞':
                            particle.append(morph.surface)
                    particles.extend(particle)

                particles = '\t'.join(particles)
                if not particles == []:
                    o_file.write(f'{verb}\t{particles}\n')

                verb = ''; v_srcs = []
                particles = []

"""
出力結果
cat verb_particle.txt | sort | uniq -c | sort -nr | head
  49 する       を
  18 する       が
  15 する       に
  14 する       と
  12 する       は      を
  10 れる       と
   9 よる       に
   8 行う       を
   8 する
   6 基づく     に

cat verb_particle.txt | grep '行う' | sort | uniq -c | sort -nr | head -n 10
   8 行う       を
   1 行う       まで    を      に
   1 行う       まで    を
   1 行う       から
   1 行う       に      により  に      を
   1 行う       は      を      をめぐって
   1 行う       で      は      て      が      に
   1 行う       て      に      は      は      は
   1 行う       に      は      で      が
   1 行う       を      に      を

cat verb_particle.txt | grep 'なる' | sort | uniq -c | sort -nr | head -n 10
   3 なる       は      に
   3 なる       が      と
   2 なる       に
   2 なる       と
   1 無くなる   は
   1 異なる     で      が
   1 異なる     も
   1 なる       から    で      は      が      て      と
   1 なる       から    で      と
   1 なる       が      にとって        と      は

cat verb_particle.txt | grep '与える' | sort | uniq -c | sort -nr | head -n 10
   1 与える     が      など    に
   1 与える     は      に      を
   1 与える     が      に
"""