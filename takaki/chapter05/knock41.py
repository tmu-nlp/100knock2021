from __future__ import annotations
from dataclasses import dataclass
from knock40 import Morph
from pprint import pprint
from typing import Tuple, List
from collections import defaultdict


@dataclass
class Chunk:
    morphs: List[Morph]
    dst   : int
    srcs  : List[int]

    def __init__(self, morphs: List[Morph], dst: int, srcs: List[int]) -> None:
        self.morphs = morphs
        self.dst    = dst
        self.srcs   = srcs

    def set_srcs(self, srcs: List[int]) -> None:
        self.srcs = srcs

    def add_morph(self, morph: Morph) -> None:
        self.morphs.append(morph)

    @staticmethod
    def parse(line) -> Tuple[int, int, Chunk]:
        line = line.split()
        cur = int(line[1])
        dst = int(line[2][:-1])
        return (cur, dst, Chunk([], dst, []))


@dataclass
class Sentence:
    chunks: List[Chunk]

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks

    @staticmethod
    def parse(lines) -> Sentence:
        chunks = []
        chunk  = Chunk([], -1, [])
        srcs   = defaultdict(lambda: [])
        for line in lines:
            if line[0] == '*':
                if len(chunk.morphs) != 0:
                    chunks.append(chunk)
                cur, dst, chunk = Chunk.parse(line)
                assert cur == len(chunks)
                if dst != -1:
                    srcs[dst].append(cur)
            else:
                chunk.add_morph(Morph.parse(line))
        if len(chunk.morphs) != 0:
            chunks.append(chunk)
        for dst, curs in srcs.items():
            chunks[dst].set_srcs(curs)
        return Sentence(chunks)


@dataclass
class Document:
    sentences: List[Sentence]

    def __init__(self, sentences: List[Sentence]) -> None:
        self.sentences = sentences

    @staticmethod
    def parse(lines) -> Document:
        sentences = []
        lns = []
        for i, line in enumerate(lines):
            if  line.strip() == 'EOS':
                if len(lns) != 0:
                    sentences.append(Sentence.parse(lns))
                    lns = []
            else:
                lns.append(line)
        return Document(sentences)


if __name__ == '__main__':
    # ----------
    with open('./ai.ja.txt.parsed', 'r') as f:
        lines = f.readlines()
    d = Document.parse(lines)
    sentence = d.sentences[1]
    # ----------
    for chunk in sentence.chunks:
        pprint(chunk)
