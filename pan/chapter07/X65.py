# 65. アナロジータスクでの正解率
# 64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．

if __name__ == '__main__':
    sys_cor = 0 # syntactic analogyの正解数
    sys_sum = 0 # syntactic analogyの問題数
    sem_cor = 0 # semantic analogyの正解数
    sem_sum = 0 # semantic analogyの問題数
    with open('/users/kcnco/github/100knock2021/pan/chapter07/result.txt') as input_file:
        for line in input_file:
            words = line.strip().split()
            if len(words) == 2:
                if 'gram' in words[1]:
                    # : gram1-adjective-to-adverb のような行
                    is_syn = True
                else:
                    # : capital-common-countries のような行
                    is_syn = False
                continue
            if is_syn:
                sys_sum += 1
                # words[3] := 正解単語、words[4] := 予測単語
                if words[3] == words[4]:
                    sys_cor += 1
            else:
                sem_sum += 1
                # words[3] := 正解単語、words[4] := 予測単語
                if words[3] == words[4]:
                    sem_cor += 1

    # 正解率の計算
    acc_syn = sys_cor /sys_sum
    acc_sem = sem_cor / sem_sum

    # 結果を表示する
    print(f'Accuracy of syntactic task: {acc_syn}')
    print(f'Accuracy of semantic task : {acc_sem}')