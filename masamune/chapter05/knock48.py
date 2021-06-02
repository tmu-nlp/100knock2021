from knock41 import sentences

def dfs(sentence, chunk, result):
    if chunk.dst == -1:
        return result
    result.append(sentence[chunk.dst])
    return dfs(sentence, sentence[chunk.dst], result)

for sentence in sentences:
    for chunk in sentence:
        n_flg = False

        for morph in chunk.morphs:
            if morph.pos == '名詞':
                n_flg = True
        if not n_flg:
            continue
        result = dfs(sentence, chunk, [chunk])
        ans = []
        for chunk in result:
            surface = ""
            for morph in chunk.morphs:
                surface += morph.surface
            ans.append(surface)
        ans = " -> ".join(ans)
        print(ans)