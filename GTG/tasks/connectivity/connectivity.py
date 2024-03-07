from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import BoolEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'connectivity'
_num_sample = 0
_num_yes = 0


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
    if g.is_directed():
        ques_str = "Is there a directed path from \
node {} to node {}?".format(
        NID(ques['start']), NID(ques['end']))
    else:
        ques_str = "Is there a path between \
node {} and node {}?".format(
        NID(ques['start']), NID(ques['end']))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results. Use the depth-first search (DFS) algorithm to detect a path:\n"
    steps_str = "Let's solve it step by step. We can use the depth-first search (DFS) algorithm to detect connectivity between two nodes.\n"

    v = ques['start']
    visited = []
    stack = []
    stack.append(v)
    # steps_str += "visited: {}, stack: {}.\n".format(NID(visited), NID(stack))
    while len(stack) != 0:
        v = stack.pop()
        if v not in visited:
            visited.append(v)
            # steps_str += "Visit node {}.\n".format(NID(v))
            for u in g.neighbors(v):
                stack.append(u)
            # steps_str += "stack: {}.\n".format(NID(stack))
    
    # ans = visited  # the node with largest page rank
    # ans_str = NID(visited)
    # steps_str += "The sequence of traversal: {}.\n".format(NID(visited))

    steps_str += "The DFS traversal start from node {} is {}.\n".format(
        NID(ques['start']), NID(visited)
    )

    ans = ques['end'] in visited
    if ans:
        ans_str = 'Yes'
        steps_str += "Node {} is in the traversal, ".format(NID(ques['end']))
        # steps_str += "Node {} is in the traversal, \
# so there is a path from node {} to node {}.".format(NID(ques['end']), NID(ques['start']), NID(ques['end']))
    else:
        ans_str = 'No'
        steps_str += "Node {} is not in the traversal, ".format(NID(ques['end']))
        # steps_str += "Node {} is in not the traversal, \
# so there is not a path from node {} to node {}.".format(NID(ques['end']), NID(ques['start']), NID(ques['end']))
    steps_str += "so the answer is "
    reject = False
    if ans:
        global _num_sample
        global _num_yes
        if _num_yes / (_num_sample + 1) > 0.5:
            reject = True
        else:
            _num_yes += 1
        # yes ans around 50%
    
    # print(steps_str)
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    false_ans = []

    choi_str = "[Yes, No]"
    if ans:
        label_str = '0'
    else:
        label_str = '1'
    
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


class Evaluator(BoolEvaluator):

    def __init__(self):
        super().__init__()
