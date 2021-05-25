cnt = 0
with open("popular-names.txt") as f:
    for line in f:
        cnt += 1
print(cnt)

#wc -l popular-names.txt
#word count -line