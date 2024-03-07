from GTG.utils.utils import NID

import re

class Mapping:

    def __init__(self, s):
        l = re.findall("<(\d+)>", s)
        if len(l) == 0:
            self.d = {}
        else:
            l = {int(x) for x in l}
            num_nodes = max(l) + 1
            
            self.d = {NID(u):str(u) for u in range(num_nodes)}

    def __call__(self, s):
        for key, value in self.d.items():
            s = s.replace(key, value)
        return s
