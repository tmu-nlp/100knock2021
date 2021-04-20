#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python3

def solve(list):
    return [sum(list, [])]

print(solve([[0, 1], [2, 3], [4, 5]]))
