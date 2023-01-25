import random

def random_matching(data):
    n = len(data)
    rm = []
    for i in range(n):
        p1 = random.choice(data)
        p2 = random.choice(data)
        if p1 != p2:
            rm.append((p1[0], p2[1]))
            rm.append((p2[0], p1[1]))
    return rm

def random_matching_triplet(data):
    anchor_message = []
    anchor_condition = []
    n = len(data)
    for i in range(n):
        p1 = random.choice(data)
        p2 = random.choice(data)
        if p1 != p2:
            anchor_condition.append((p1[0], p1[1], p2[1]))
            anchor_message.append((p1[1], p1[0], p2[0]))
    return anchor_condition, anchor_message