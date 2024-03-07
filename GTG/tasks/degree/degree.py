from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import IntEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'degree'
_num_sample = 0
_num_zero = 0


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    node = random.randint(0, num_nodes - 1)    # return random integer in range [a, b]
    
    ques = {
        'u': node
    }
    if g.is_directed():
        ques_str = "What is the out-degree of node {}?".format(
            NID(node))
    else:
        ques_str = "What is the degree of node {}?".format(
            NID(node))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of calculating the degree of node {} is:\n".format(
    #     NID(node))
    steps_str = "Let's solve this problem step by step.\n"
    
    node = ques['u']
    u_nei = list(g.neighbors(node))
    if g.is_directed():
        steps_str += "The successors of node {} are: {}, ".format(
            NID(node), NID(u_nei)
        )
        steps_str += "and there are {:d} successor nodes.\n".format(len(u_nei))
    else:
        steps_str += "The neighbors of node {} are: {}, ".format(
            NID(node), NID(u_nei)
        )
        steps_str += "and there are {:d} neighbor nodes.\n".format(len(u_nei))
    steps_str += "So the degree of node {} is ".format(NID(node))
    
    ans = len(u_nei)
    ans_str = str(ans)
    reject = False
    if ans == 0:
        global _num_sample
        global _num_zero
        # zero ans no more than 20%
        if _num_zero / _num_sample > 0.2:
            reject = True
        else:
            _num_zero += 1

    # print(steps_str)
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    num_choices = 4
    
    false_ans = set()
    if ans != 0:
        false_ans.add(0)
    while len(false_ans) < num_choices - 1:
        x = random.randint(0, 10)
        if x == ans:
            continue
        false_ans.add(x)
    
    choi = np.array(list(false_ans) + [ans])
    np.random.shuffle(choi)
    label_str = str(np.arange(num_choices)[choi == ans][0])
    choi_str = "[" + ", ".join(["{:d}".format(x) for x in choi]) + "]"

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


class Evaluator(IntEvaluator):

    def __init__(self):
        super().__init__()
