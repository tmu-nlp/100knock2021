from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
fr = open('questions-words.txt', 'r')
fw = open('ans-knock64.txt', 'w')
i = 0

for line in fr:
    line = line.strip()
    l = line.split(' ')

    if l[0] == ':':
        category = l[1]
    else:
        word, value = model.most_similar(positive=[l[1], l[2]], negative=[l[0]], topn = 1)[0]
        fw.write(category + ' ' + line + ' ' + word + ' ' + str(value) + '\n')

fr.close()
fw.close()


'''

時間かかったので途中で切りました
questions-words.txt 19558行
ans-knock64.txt 12357行

head -10 './ans-knock64.txt'

capital-common-countries Athens Greece Baghdad Iraq Iraqi 0.6351870894432068
capital-common-countries Athens Greece Bangkok Thailand Thailand 0.7137669324874878
capital-common-countries Athens Greece Beijing China China 0.7235777974128723
capital-common-countries Athens Greece Berlin Germany Germany 0.6734622120857239
capital-common-countries Athens Greece Bern Switzerland Switzerland 0.4919748306274414
capital-common-countries Athens Greece Cairo Egypt Egypt 0.7527809739112854
capital-common-countries Athens Greece Canberra Australia Australia 0.583732545375824
capital-common-countries Athens Greece Hanoi Vietnam Viet_Nam 0.6276341676712036
capital-common-countries Athens Greece Havana Cuba Cuba 0.6460992097854614
capital-common-countries Athens Greece Helsinki Finland Finland 0.6899983882904053

'''