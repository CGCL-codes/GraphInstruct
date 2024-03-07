from GTG.utils.utils import NID, graph_generation, generate_random_directed_acyclic_graph, make_sample
from GTG.utils.evaluation import BoolEvaluator

import networkx as nx
import random
import numpy as np
from collections import deque


TASK_NAME = 'cycle'
_num_sample = 0
_num_not_have_cycle = 0


def dict_formatter(d):
    # key is node id
    l = ['node {}: {}'.format(NID(k), d[k]) for k in d]
    s = '{' + ', '.join(l) + '}'
    return s


def question_generation(config, g):
    ques = {}

    if g.is_directed():
        ques_str = "Does the graph have a cycle? For a directed graph, a cycle is a closed path that traverses through a sequence of nodes and directed edges, eventually returning to the starting node. "
    else:
        ques_str = "Does the graph have a cycle? For an undirected graph, a cycle is a closed path that traverses through a sequence of nodes and edges, eventually returning to the starting node. "

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
#     steps_str = "Reasoning process and intermediate results of \
# checking if the graph has a cycle (use topological sorting):\n"
    steps_str = "Let's solve it step by step. We can use the topological sorting algorithm to detect a cycle in the graph.\n"

    in_degree = {node: 0 for node in g}
    for node in g:
        for neighbor in g.neighbors(node):
            in_degree[neighbor] += 1
    # steps_str += "In-degree of all the nodes: {}.\n".format(
    #     dict_formatter(in_degree)
    # )

    queue = deque([node for node in g if in_degree[node] == 0])
    result = []

    while len(queue):
        steps_str += "Nodes with 0 in-degree: {}.\n".format(NID(list(queue)))
        node = queue.popleft()
        result.append(node)
        steps_str += "Visit node {} and remove it from the graph.\n".format(NID(node))

        for neighbor in g.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        
        # steps_str += "Remove node {} from the graph.\n".format(NID(node))
        # steps_str += "In-degree of all the nodes: {}.\n\n".format(
        #     dict_formatter(in_degree)
        # )
    
    steps_str += "The result of topological sorting: {} ".format(NID(result))
    if len(result) != g.number_of_nodes():
        ans = True
        ans_str = 'Yes'
        # steps_str += "does not contain all the nodes in the graph, so the graph has a cycle."
        steps_str += "does not contain all the nodes in the graph, so the answer is "
    else:
        ans = False
        ans_str = 'No'
        # steps_str += "contains all the nodes in the graph, so the graph does not have a cycle."
        steps_str += "contains all the nodes in the graph, so the answer is "

    # print(steps_str)
    return steps_str, ans, ans_str


def _answer_and_inference_steps_generation(config, g, ques):
#     steps_str = "Reasoning process and intermediate results of \
# checking if the graph has a cycle (use topological sorting):\n"
    steps_str = "We can use the topological sorting algorithm to detect a cycle in the graph.\n"

    in_degree = {node: 0 for node in g}
    for node in g:
        for neighbor in g.neighbors(node):
            in_degree[neighbor] += 1
    steps_str += "In-degree of all the nodes: {}.\n".format(
        dict_formatter(in_degree)
    )

    queue = deque([node for node in g if in_degree[node] == 0])
    result = []

    while len(queue):
        steps_str += "Nodes with 0 in-degree: {}.\n".format(NID(list(queue)))
        node = queue.popleft()
        result.append(node)
        steps_str += "Add node {} to results list.\n".format(NID(node))

        for neighbor in g.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        
        steps_str += "Remove node {} from the graph.\n".format(NID(node))
        steps_str += "In-degree of all the nodes: {}.\n\n".format(
            dict_formatter(in_degree)
        )
    
    steps_str += "The result of topological sorting: {} ".format(NID(result))
    if len(result) != g.number_of_nodes():
        ans = True
        ans_str = 'Yes'
        # steps_str += "does not contain all the nodes in the graph, so the graph has a cycle."
        steps_str += "does not contain all the nodes in the graph, so the answer is "
    else:
        ans = False
        ans_str = 'No'
        # steps_str += "contains all the nodes in the graph, so the graph does not have a cycle."
        steps_str += "contains all the nodes in the graph, so the answer is "

    # print(steps_str)
    return steps_str, ans, ans_str


def choices_generation(config, g, ques, ans):
    false_ans = []

    choi_str = "[Yes, No]"
    if ans:
        label_str = '0'
    else:
        label_str = '1'
    
    return choi_str, label_str


def generate_a_sample(config):
    global _num_sample
    global _num_not_have_cycle
    if _num_not_have_cycle / (_num_sample + 1) < 0.5:
        g = generate_random_directed_acyclic_graph(config)
    else:
        g = graph_generation(config)
    
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, ques)
    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    _num_sample += 1
    if not ans:
        _num_not_have_cycle += 1
    return sample


class Evaluator(BoolEvaluator):

    def __init__(self):
        super().__init__()
