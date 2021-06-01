#####
#与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，Graphviz等を用いるとよい．
from graphviz import Digraph
from X43 import analyze_chunk

def make_graph_graphviz(nodes):
    #有向グラフのオブジェクトを作成（pngファイルで保存）
    G = Digraph(format='png')
    #ノードの形を円形に指定
    G.attr('node', shape='circle')

    for node in nodes:
        #ノードに係り元を追加
        G.node(node[0])
        #ノードに係り先を追加
        G.node(node[1])
        #係り元→係り先のエッジを追加
        G.edge(node[0], node[1])

    #graph_44というファイル名で保存し、pngファイルを表示
    G.render('graph_44', view=True)

ai_chunks = analyze_chunk('/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed')[0]
# （係り元, 係り先）のタプルが入るリスト
nodes = []

for ai_chunk in ai_chunks:
    if ai_chunk.dst != -1:
        s = ai_chunk.join_surface_womark()
        d = ai_chunks[ai_chunk.dst].join_surface_womark()

        if s != '' and d != '':
            nodes.append((s, d))

# nodesを入れて関数を実行
make_graph_graphviz(nodes)
