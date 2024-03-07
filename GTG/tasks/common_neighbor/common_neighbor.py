from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import IntEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'common_neighbor'
_num_sample = 0
_num_sample_zero_cn = 0


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    start = random.randint(0, num_nodes - 1)    # return random integer in range [a, b]
    end = random.randint(0, num_nodes - 1)
    while start == end:
        end = random.randint(0, num_nodes - 1)
    
    ques = {
        'u': start, 
        'v': end
    }
    
    ques_str = "Calculate the number of common neighbors of \
node {} and node {}. ".format(NID(ques['u']), NID(ques['v']))

    if g.is_directed():
        ques_str += "In the context of a directed graph, we consider a node's successors as its neighbors. " 

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of \
# calculating the common neighbors of node {} and node {}:\n".format(
    # NID(ques['u']), NID(ques['v']))
    steps_str = "Let's calulate the number of common neighbors step by step.\n"
    
    u = ques['u']
    v = ques['v']
    u_nei = list(g.neighbors(u))
    v_nei = list(g.neighbors(v))

    steps_str += "Fisrtly, the neighbors of node {} are {}.\n".format(
        NID(u), NID(np.array(u_nei))
    )
    steps_str += "Secondly, the neighbors of node {} are {}.\n".format(
        NID(v), NID(np.array(v_nei))
    )
    cn = np.array(list(set(u_nei) & set(v_nei)))
    # print(cn)
    steps_str += "Common neighbors of \
node {} and node {}: {}, which contains {} nodes.\n".format(NID(u), NID(v), NID(cn), len(cn))
    
    ans = len(cn)
    ans_str = str(ans)
    # steps_str += "So the number of common neighbors is {}.".format(ans)
    steps_str += "So the number of common neighbors is "
    reject = False
    if ans == 0:
        global _num_sample
        global _num_sample_zero_cn
        if _num_sample_zero_cn / (_num_sample + 1) > 0.2:
            reject = True
        else:
            _num_sample_zero_cn += 1
        # zero ans no more than 20%

    # print(steps_str)
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    num_choices = 4
    
    false_ans = set()
    if ans != 0:
        false_ans.add(0)
    while len(false_ans) < num_choices - 1:
        x = random.randint(1, max(10, ans * 2))
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
