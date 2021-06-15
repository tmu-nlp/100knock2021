

with open('ans.txt','r') as f:
    sem_count=0
    sem_correct=0
    syn_count=0
    syn_correct=0

    for line in f:
        line=line.split(' ')
        if not line[0].startswith('gram'):
            sem_count+=1
            if line[4]==line[5]:
                sem_correct+=1
        else:
            syn_count+=1
            if line[4]==line[5]:
                syn_correct+=1
        
print("{} {} {} {}".format(syn_correct,syn_count,sem_correct,sem_count))
semantic_analogy=sem_correct/sem_count
print(semantic_analogy)
syntactic_analogy=syn_correct/syn_count
print(syntactic_analogy)
print("semantic analogy={}\n".format(semantic_analogy))
print("syntactic analogy={}\n".format(syntactic_analogy))
