import sys

def create_file_generater(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f:
            yield line

n = int(sys.argv[1])
g1 = create_file_generater("data/popular-names.txt")
l = 0
with open("data/popular-names.txt", "r", encoding = "utf-8") as f:
    l = len([i for i in f])
x = l // n
if l % n != 0:
    x += 1
for i in range(n):
    with open (f"result/knock16_{i}.txt", "w",encoding = "utf-8") as f:
        for j in range(x):
            try:
                f.write(f"{next(g1).strip()}\n")
            except:
                pass