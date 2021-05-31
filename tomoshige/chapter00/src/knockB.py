def knockB_short(_2d_list):
    result = sum(_2d_list, [])
    return result


def knockB_efficient(_2d_list):
    import itertools
    flatten_list = itertools.chain.from_iterable(_2d_list)
    return flatten_list


def knockB_versatile(nested_list):
    import collections
    def flatten(l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
    return flatten(nested_list)


def run_all(func_names, *args):
    print("-" * 23)
    print("[+] Input")
    print(*args)
    print()
    for func_name in func_names:
        print("[x]", func_name)
        print(list(eval(f"{func_name}(*args)")))
        print()


if __name__ == "__main__":
    _2d_list = [[0, 1], [2, 3], [4, 5]]
    nested_list = [[0, 1], [2, 3, [4, [5, [range(6, 8)]]]]]
    func_names = [e for e in dir() if e.startswith("knockB")]
    run_all(func_names, _2d_list)
    run_all(func_names, nested_list)
