from GTG.utils.utils import NID, NID_edges, generate_bipartite_graph, make_sample
from GTG.utils.parse_output import parse_edge_list
from GTG.utils.evaluation import BaseEvaluator

import networkx as nx
import random
import numpy as np
from collections import deque


TASK_NAME = 'bipartite'


def dict_formatter(d):
    # key is node id
    l = ['{}: {}'.format(NID(k), NID(d[k])) for k in d]
    s = '{' + ', '.join(l) + '}'
    return s


def question_generation(config, g, n1, n2):
    ques_str = "Find a maximum matching in the bipartite graph.\n"
    ques_str += "Nodes set 1 contains:"
    for i in range(n1):
        ques_str += " " + NID(i)
    ques_str += ".\n"
    ques_str += "Nodes set 2 contains:"
    for i in range(n1, n1 + n2):
        ques_str += " " + NID(i)
    ques_str += "."

    return ques_str


def answer_and_inference_steps_generation(config, g, n1, n2):
    steps = ""
    
    # def hungarian_algorithm(g, n1, n2):
    # Initialize an empty matching
    steps += "To find a maximum matching in the bipartite graph, let's run the Hungarian algorithm step by step.\n"
    steps += "Firstly, initialize an empty matching {}.\n"
    matching = {}
    
    def dfs(node):
        if node in visited:
            return False
        visited.add(node)
        
        for neighbor in g.neighbors(node):
            if neighbor not in matching or dfs(matching[neighbor]):
                matching[neighbor] = node
                return True
        
        return False
    
    steps += "Iterate over nodes in set 1:\n"
    for node in range(n1):
        steps += "Search from node {}. ".format(NID(node))
        visited = set()
        dfs(node)
        steps += "Updated matching: {}.\n".format(dict_formatter(matching))
    
    # Retrieve the maximum matching
    max_matching = [(matching[node], node) for node in matching]
    X = np.array(max_matching)
    max_matching = X[X[:,0].argsort()]

    steps += "So the maximum matching is "

    ans = max_matching
    ans_str = NID_edges(ans)

    # print(steps_str)
    return steps, ans, ans_str


def choices_generation(config, g, ques, ans):
    pass


def generate_a_sample(config):
    g, n1, n2 = generate_bipartite_graph(config)
    ques_str = question_generation(config, g, n1, n2)  ###
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, n1, n2)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str
    )
    sample['n1'] = str(n1)
    sample['n2'] = str(n2)
    return sample


class Evaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        output_ans = parse_edge_list(output_ans_str, sample)
        # ensure valid node id in output_ans
        return output_ans

    def check_correctness(self, sample, output_ans):
        ans = parse_edge_list(sample['answer'])
        if len(output_ans) != len(ans):
            return False

        ques = sample['question']
        
        tmp = ques[ques.find('Nodes set 1 contains: ') + len('Nodes set 1 contains: '):]
        node_set_1 = set(tmp[:tmp.find('.')].split(' '))

        tmp = ques[ques.find('Nodes set 2 contains: ') + len('Nodes set 2 contains: '):]
        node_set_2 = set(tmp[:tmp.find('.')].split(' '))

        set1_nodes = set()
        set2_nodes = set()
        for e in output_ans:
            if e[0] in node_set_1 and e[1] in node_set_2:
                pass
            elif e[0] in node_set_2 and e[1] in node_set_1:
                e = (e[1], e[0])
            else:
                return False
            
            if e[0] in set1_nodes:
                return False
            set1_nodes.add(e[0])

            if e[1] in set2_nodes:
                return False
            set2_nodes.add(e[1])
        return True
