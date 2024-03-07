from GTG.utils.utils import NID, NID_edges, graph_generation, make_sample
from GTG.utils.evaluation import FloatEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'clustering_coefficient'
_num_sample = 0
_num_zero = 0


def question_generation(config, g):

    num_nodes = g.number_of_nodes()
    node = random.randint(0, num_nodes - 1)    # return random integer in range [a, b]
    
    ques = {
        'u': node
    }
    ques_str = "What is the clustering coefficient of node {}?".format(NID(node))
    if g.is_directed():
        ques_str += " For a directed graph, we consider a node's successors as its neighbors."
    
    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques, directed):
    steps_str = "Let's calculate the clustering coefficient step by step.\n"
    
    if g.is_directed():
        steps_str += "For a directed graph, the clustering coefficient for a node u is T / (D * (D - 1)), \
where T is the number of edges between neighbors of u, and D is the out-degree of u.\n"
    else:
        steps_str += "For an undirected graph, the clustering coefficient for a node u is 2 * T / (D * (D - 1)), \
where T is the number of edges between neighbors of u, and D is the degree of u.\n"
    
    u = ques['u']
    nei = list(g.neighbors(u))
    deg = len(nei)

    if deg <= 1:
        reject = True
        return None, None, None, reject

    edge_list = []
    nu = 0
    if g.is_directed():
        for i in nei:
            for j in nei:
                if g.has_edge(i, j):
                    nu += 1
                    edge_list.append((i, j))
        cc = nu / (deg * (deg - 1))
    else:
        for idx, i in enumerate(nei):
            for j in nei[idx + 1:]:
                if g.has_edge(i, j):
                    nu += 1
                    edge_list.append((i, j))
        cc = 2 * nu / (deg * (deg - 1))

    if g.is_directed():
        steps_str += "Node {}'s neighbors are {}. There are {} edges between them: {}.\n".format(
            NID(u), NID(nei), nu, NID_edges(edge_list)
        )
        steps_str += "Node {}'s out-degree is {}.\n".format(NID(u), deg)
        steps_str+="So the the clustering coefficient of node {} is {} / ({} * ({} - 1)) = ".format(
            NID(ques['u']), nu, deg, deg
        )
    else:
        steps_str += "Node {}'s neighbors are {}. There are {} edges between them: {}.\n".format(
            NID(u), NID(nei), nu, NID_edges(edge_list)
        )
        steps_str += "Node {}'s degree is {}.\n".format(NID(u), deg)
        steps_str+="So the the clustering coefficient of node {} is 2 * {} / ({} * ({} - 1)) = ".format(
            NID(ques['u']), nu, deg, deg
        )
    ans = cc
    ans_str = "{:.4f}".format(ans)

    reject = False
    if ans < 0.00001:
        global _num_sample
        global _num_zero
        # zero ans no more than 20%
        if _num_zero / (_num_sample + 1) > 0.2:
            reject = True
        else:
            _num_zero += 1
    
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    num_choices = 4
    
    false_ans = set()
    if ans != 0:
        false_ans.add(0)
    while len(false_ans) < num_choices - 1:
        x = random.uniform(0, 1)
        if abs(x - ans) / (ans + 1e-6) < 0.05:
            continue
        false_ans.add(x)
    
    choi = np.array(list(false_ans) + [ans])
    np.random.shuffle(choi)
    label_str = str(np.arange(num_choices)[choi == ans][0])
    choi_str = "[" + ", ".join(["{:.4f}".format(x) for x in choi]) + "]"

    return choi_str, label_str


def generate_a_sample(config):
    g = graph_generation(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques, g.is_directed())
    
    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques, g.is_directed())
    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    
    global _num_sample
    _num_sample += 1
    return sample


class Evaluator(FloatEvaluator):

    def __init__(self):
        super().__init__()
