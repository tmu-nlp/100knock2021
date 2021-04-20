#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p python3

def solve(list):
    return dict(enumerate(list))

print(solve(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
