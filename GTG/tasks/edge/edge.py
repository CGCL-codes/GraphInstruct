from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import BoolEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'edge'
_num_sample = 0
_num_no = 0


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
        ques_str = "Is there a directed edge from node {} to node {}?".format(
            NID(ques['start']), NID(ques['end']))
    else:
        ques_str = "Is there an edge between node {} and node {}?".format(
            NID(ques['start']), NID(ques['end']))
    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    steps_str = "Let's do it step by step.\n"
    
    u_nei = list(g.neighbors(ques['start']))
    if g.is_directed():
        dev_name = "successors"
    else:
        dev_name = "neighbors"
    steps_str += "The {} of node {} are: {}".format(dev_name, NID(ques['start']), NID(u_nei))
    if ques['end'] in u_nei:
        ans = True
        ans_str = 'Yes'
        steps_str += ", which contains node {}. ".format(NID(ques['end']))
        steps_str += "So the answer is "
    else:
        ans = False
        ans_str = 'No'
        steps_str += ", which does not contain node {}. ".format(NID(ques['end']))
        steps_str += "So the answer is "
    
    reject = False
    if ans == False:
        global _num_sample
        global _num_no
        # ans no no more than 20%
        if _num_no / (_num_sample + 1) > 0.5:
            reject = True
        else:
            _num_no += 1

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
