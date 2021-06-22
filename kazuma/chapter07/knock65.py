from knock60 import load_vectors
from pprint import pprint

def knock65():
    with open("data/result_knock64.txt", "r") as f1:
        is_semantic = None
        sem_tot = 0
        sem_cor = 0
        syn_tot = 0
        syn_cor = 0
        for line in f1:
            words = line.strip().split(" ")
            if words[0][0] == ":":
                if words[1][:4] == "gram":
                    is_semantic = False
                else:
                    is_semantic = True
                continue
            plus_num = 0
            if words[3] == words[4]:
                plus_num = 1
            if is_semantic:
                sem_tot += 1
                sem_cor += plus_num
            else:
                syn_tot += 1
                syn_cor += plus_num

        print(sem_tot,sem_cor,syn_tot,syn_cor)
        print("意味的アナロジー:", sem_cor/sem_tot)
        print("文法的アナロジー:", syn_cor/syn_tot)

if __name__ == "__main__":
    knock65()