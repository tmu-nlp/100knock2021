with open('64.txt') as f:
    sem_cnt, sem_cor, syn_cnt, syn_cor = 0, 0, 0, 0
    now_category = ''
    for line in f:
        line = line.split()
        if len(line) == 2:
            now_category = line[1]
            continue
        if now_category[:4] == 'gram':
            syn_cnt += 1
            if line[3] == line[4]:
                syn_cor += 1
        else:
            sem_cnt += 1
            if line[3] == line[4]:
                sem_cor += 1
    
    print(f'意味的アナロジー：{sem_cor/sem_cnt:.3f}')
    print(f'文法的アナロジー：{syn_cor/syn_cnt:.3f}')

'''
意味的アナロジー：0.731
文法的アナロジー：0.740
'''
