#方向なしグラフのモジュールはこっち
#from graphviz import Graph
#方向ありグラフのモジュール
from graphviz import Digraph
from knock41 import phrases

g = Digraph(format = 'png')
g.attr('node', shape='box', fontname='MS Gothic')

def make_edge(phrase):
	for idx, ch in enumerate(phrase):
		if int(ch.dst) != -1:
			g.node(ch.phrase)
	for idx, ch in enumerate(phrase):
		g.edge(ch.phrase, phrase[int(ch.dst)].phrase)

make_edge(phrases[0])
g.render('./knock44')