import CaboCha

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

class Chunk:
    def __init__(self, sentence, morphs, dst, srcs):
        self.sentence = sentence
        self.morphs = morphs
        self.dst    = dst
        self.srcs   = srcs
