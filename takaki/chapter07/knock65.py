with open('./tmp/knock64.txt') as f:
    lines = f.readlines()

sem_cnt = 0
sem_cor = 0
syn_cnt = 0
syn_cor = 0

for line in lines:
    words = line.split()
    if line[0].startswith('gram'):
        syn_cnt += 1
        if line[4] == line[5]:
            syn_cor += 1
    else:
        sem_cnt += 1
        if line[4] == line[5]:
            sem_cor += 1

print(f'semantic analogy  : {sem_cor/sem_cnt}')
print(f'syntactic analogy : {syn_cor/syn_cnt}')
