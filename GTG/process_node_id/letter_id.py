from GTG.utils.utils import NID

import random
import re


def get_rand_letter(n=3):
    # random letter from a to z
    return ''.join([chr(random.randint(ord('A'), ord('Z'))) for _ in range(n)])


class Mapping:

    def __init__(self, s):
        l = re.findall("<(\d+)>", s)
        if len(l) == 0:
            self.d = {}
        else:
            l = {int(x) for x in l}
            num_nodes = max(l) + 1

        names = set()
        while len(names) < num_nodes:
            names.add(get_rand_letter(3))
        names = list(names)

        self.d = {NID(u):names[u] for u in range(num_nodes)}

    def __call__(self, s):
        for key, value in self.d.items():
            s = s.replace(key, value)
        return s
