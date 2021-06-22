
with open('./questions-words-re.txt') as f:
    sem_cnt = 0; sem_t = 0 #cnt: データ数, t: 一致数
    syn_cnt = 0; syn_t = 0
    flg = None
    for line in f:
        line = line.split()
        if len(line) == 1:
            if 'gram' in line[0]: #カテゴリーにgramが入っているとsemantic analogy
                flg = True 
            else:              #それ以外はsyntactic analogy
                flg = False
            continue

        if flg:
            syn_cnt += 1
            if line[3] == line[4]:
                syn_t += 1
        
        else:
            sem_cnt += 1
            if line[3] == line[4]:
                sem_t += 1

    print(f'正解率: {sem_t/sem_cnt}（semantic analogy）')
    print(f'正解率: {syn_t/syn_cnt}（syntactic analogy）')

    '''
    出力結果
    正解率: 0.7308602999210734（semantic analogy）
    正解率: 0.7400468384074942（syntactic analogy）
    '''
