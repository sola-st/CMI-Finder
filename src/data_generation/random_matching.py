import random

def random_matching(data, n = 1000000):
    rm = []
    for i in range(n):
        p1 = random.choice(data)
        p2 = random.choice(data)
        if p1 != p2:
            rm.append((p1[0], p2[1]))
            rm.append((p2[0], p1[1]))
    return rm