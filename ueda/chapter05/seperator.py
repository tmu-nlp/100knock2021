import re
with open(r'/Users/Naoya/Downloads/ai/ai.ja.txt', encoding="utf-8") as f, open(r'/Users/Naoya/Downloads/ai/ai.jap.txt', 'w', encoding="utf-8") as g:
    for line in f:
        print(line)
        line = re.sub('。', '。\n\n', line)
        g.write(line)

#Cabochaの仕様か不具合かで、「。」のあるchunkでdst=-1が出ない。これで強制的に改行させることにより解決。