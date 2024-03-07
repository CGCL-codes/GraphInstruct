from GTG.utils.utils import NID, make_sample
from GTG.utils.utils import graph_generation_with_edge_weight
from GTG.utils.utils import get_neighbor_and_edge_weight
from GTG.utils.utils import graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str, graph_with_edge_weight_to_natural_language
from GTG.utils.evaluation import IntEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'shortest_path'


def dict_formatter(d):
    # key is node id
    l = ['node {}: {}'.format(NID(k), d[k]) for k in d]
    s = '{' + ', '.join(l) + '}'
    return s


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    start = random.randint(0, num_nodes - 1)    # return random integer in range [a, b]
    end = random.randint(0, num_nodes - 1)
    while start == end:
        end = random.randint(0, num_nodes - 1)
    
    ques = {
        'start': start, 
        'end': end
    }
    
    ques_str = "Calculate the distance of the shortest path \
from node {} to node {}.".format(
    NID(ques['start']), NID(ques['end']))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
#     steps_str = "Reasoning process and intermediate results of the Dijsktra algorithm: Starting the search process of \
# the shortest path from node {} to node {}.\n".format(
#     NID(ques['start']), NID(ques['end']))
    steps_str = "Let's solve it step by step. We can use the Dijsktra algorithm.\n"
    
    num_nodes = g.number_of_nodes()
    unvisited = {node: np.Inf for node in range(num_nodes)}
    visited = {}
    current = ques['start']
    currentDistance = 0
    unvisited[current] = currentDistance

    round = 0

    while True:
        steps_str += "Round {}:\n".format(round)
        round += 1
        steps_str += "The unvisited nodes are: {}\nThe visited nodes are: {}\n".format(
            dict_formatter(unvisited), dict_formatter(visited)
        )
        for neighbour, distance in get_neighbor_and_edge_weight(g, current):
            if neighbour not in unvisited:
                continue
            newDistance = currentDistance + distance
            if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                unvisited[neighbour] = newDistance
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited:
            break
        candidates = [node for node in unvisited.items() if node[1]]
        current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]

    steps_str += "Finally, the distances to the visited nodes are {}.\n".format(
        dict_formatter(visited)
    )
    ans = visited[ques['end']]
    ans_str = str(ans)
    # steps_str += "The shortest distance between node {} and node {} is {}.".format(
    #     NID(ques['start']), NID(ques['end']), ans
    # )
    steps_str += "So the shortest distance from node {} to node {} is ".format(
        NID(ques['start']), NID(ques['end']))

    reject = False
    if np.isinf(ans):
        reject = True
    
    # print(steps_str)
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    num_choices = 4
    
    false_ans = set()
    while len(false_ans) < num_choices - 1:
        x = random.randint(1, 
            min(g.number_of_nodes() * 10, max(10, ans * 2))
        )
        if abs(x - ans) <= 2:
            continue
        else:
            false_ans.add(x)
    
    choi = np.array(list(false_ans) + [ans])
    np.random.shuffle(choi)
    label_str = str(np.arange(num_choices)[choi == ans][0])
    choi_str = "[" + ", ".join([str(x) for x in choi]) + "]"

    return choi_str, label_str


def generate_a_sample(config):
    g = graph_generation_with_edge_weight(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    
    while reject:
        g = graph_generation_with_edge_weight(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    
    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str,
        g_str=graph_with_egde_weight_to_str(g),
        g_adj_str=graph_with_egde_weight_to_adj_str(g),
        g_adj_nl=graph_with_edge_weight_to_natural_language(g)
    )
    return sample


class Evaluator(IntEvaluator):

    def __init__(self):
        super().__init__()
