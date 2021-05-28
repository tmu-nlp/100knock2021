def knockA(seq):
    result = dict(enumerate(seq))
    return result


if __name__ == "__main__":
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(knockA(day_names))
