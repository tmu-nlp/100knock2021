import sys

n = int(sys.argv[1])
with open ("data/popular-names.txt", "r", encoding = "utf-8") as f1:
    for i in range(n):
        print(next(f1).strip())


# ↓残骸をのこしとく。意味ないことしてたわ。
# def create_file_generater(file_path):
#     with open(file_path, "r", encoding = "utf-8") as f1:
#         print(next(f1))
#         for line in f1:
#             yield line

# n = int(sys.argv[1])
# g1 = create_file_generater("data/popular-names.txt")
# for i in range(n):
#     print(next(g1).strip())
