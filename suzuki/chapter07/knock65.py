#ans-knock64.txtの5行目と6行目が一致していれば正解として
#正解率はcategory毎に正解の数をcategoryの事例数で割る

f = open('ans-knock64.txt', 'r')

sem_count = 0
sem_correct = 0
syn_count = 0
syn_correct = 0

for line in f:
    l = line.strip().split(' ')

    #意味的アナロジー
    if not l[0].startswith('gram'):
        sem_count += 1
        if l[4] == l[5]:
            sem_correct += 1
    
    #文法的アナロジー
    else:
        syn_count += 1
        if l[4] == l[5]:
            syn_correct += 1

sem_acc = sem_correct / sem_count
syn_acc = syn_correct / syn_count

print('意味的アナロジー正解率: ' + str(sem_acc))
print('文法的アナロジー正解率: ' + str(syn_acc))

f.close()

'''

questions-words.txt 19558行
ans-knock64.txt 12357行

意味的アナロジー正解率: 0.7308602999210734
文法的アナロジー正解率: 0.6152522935779816

'''