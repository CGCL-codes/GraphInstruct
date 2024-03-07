from GTG.utils.utils import NID, NID_edges, graph_generation, make_sample
from GTG.utils.evaluation import NodeSetEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'neighbor'
_num_sample = 0
_num_zero = 0


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    u = random.randint(0, num_nodes - 1)
    
    ques = {
        'u': u, 
    }
    ques_str = "Which are the neighbor nodes of node {}? ".format(
        NID(ques['u']))
    if g.is_directed():
        ques_str += "In the context of a directed graph, we consider a node's successors as its neighbors. " 

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    steps_str = "Let's solve it step by step.\n"
    
    u = ques['u']
    u_nei = list(g.neighbors(u))
    us = np.full(len(u_nei), fill_value=u)
    edges = np.stack([us, u_nei]).transpose()   
     
    # if g.is_directed():
    #     steps_str += "The edges in which node {} is the source node are: {}. ".format(
    #         NID(u), NID_edges(edges))
    # else:
    #     steps_str += "The edges which contain node {} are: {}. ".format(
    #         NID(u), NID_edges(edges))
    # steps_str += "So the neighbors of node {} are ".format(NID(u))
    steps_str += "Node {} connects to nodes {}, so the neighbors of node {} are ".format(
        NID(u), NID(u_nei), NID(u)
    )

    ans = u_nei
    ans_str = NID(u_nei)

    reject = False
    if len(ans) == 0:
        global _num_sample
        global _num_zero
        if _num_zero / (_num_sample + 1) > 0.1:
            reject = True
        else:
            _num_zero += 1
    elif len(ans) >= g.number_of_nodes() - 1:
        reject = True

    # print(steps_str)
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    false_ans = []

    def get_random_false():
        k = random.randint(0, g.number_of_nodes() - 1)
        if len(ans) == 0:
            while k == 0:
                k = random.randint(0, g.number_of_nodes() - 1)
        all_nodes = np.arange(g.number_of_nodes())
        np.random.shuffle(all_nodes)
        x = all_nodes[:k]
        while set(x) == set(ans):
            np.random.shuffle(all_nodes)
            x = all_nodes[:k]
        return x

    for _ in range(3):
        false_ans.append(get_random_false())

    _choi_list = [ans] + false_ans
    choi_list = []
    ind = np.arange(4)
    np.random.shuffle(ind)
    for i in ind:
        choi_list.append(NID(_choi_list[i]))
    label_str = str(np.arange(4)[ind == 0][0])
    choi_str = "[" + ", ".join(choi_list) + "]"

    return choi_str, label_str


def generate_a_sample(config):
    g = graph_generation(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    
    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    
    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    global _num_sample
    _num_sample += 1
    return sample


class Evaluator(NodeSetEvaluator):

    def __init__(self):
        super().__init__()
