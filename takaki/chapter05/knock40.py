from __future__ import annotations
from dataclasses import dataclass
from pprint import pprint
from typing import List


@dataclass
class Morph:
    surface: str
    base   : str
    pos    : str
    pos1   : str

    def __init__(self, surface: str, base: str, pos: str, pos1: str) -> None:
        self.surface = surface
        self.base    = base
        self.pos     = pos
        self.pos1    = pos1

    @staticmethod
    def parse(line) -> Morph:
        surface, data = line.split()
        data = data.split(',')
        return Morph(surface, data[6], data[0], data[1])


def parse(lines) -> List[[Morph]]:
    sentences, morphs = [], []
    for line in lines:
        if line[0] == '*':
            continue
        elif line.strip() == 'EOS':
            if len(morphs) != 0:
                sentences.append(morphs)
                morphs = []
            continue
        else:
            morphs.append(Morph.parse(line))
    return sentences



if __name__ == '__main__':
    with open('./ai.ja.txt.parsed', 'r') as f:
        lines = f.readlines()
    sentences = parse(lines)
    pprint(sentences[1])
