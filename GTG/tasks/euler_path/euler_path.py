from GTG.utils.utils import NID, make_sample, generate_Euler_path_graph, edge_list_str_to_graph
from GTG.utils.evaluation import NodeListEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'euler_path'


def generate_a_sample(config):
    g, path = generate_Euler_path_graph(config)
    ans_str = NID(path)

    ques_str = "Find a Euler path in this graph. An Euler path in a graph is a path that traverses each edge exactly once, but not necessarily every node. The starting node and ending node may be different. "

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str
    )
    return sample


class Evaluator(NodeListEvaluator):

    def __init__(self):
        super().__init__()

    def check_correctness(self, sample, output_ans):
        g = edge_list_str_to_graph(sample['graph'], directed=sample['directed'])

        path = output_ans
        if len(path) - 1 != g.number_of_edges():
            return False
        
        edge_set = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not g.has_edge(u, v):
                return False
            if g.is_directed():
                e = (u, v)
                if e in edge_set:
                    return False
                edge_set.add(e)
            else:
                e1 = (u, v)
                e2 = (v, u)
                if e1 in edge_set or e2 in edge_set:
                    return False
                edge_set.add(e1)
                edge_set.add(e2)
        
        return True
