from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import FloatEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'Jaccard'
_num_sample = 0
_num_sample_zero_cn = 0


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    start = random.randint(0, num_nodes - 1)    # return random integer in range [a, b]
    end = end = random.randint(0, num_nodes - 1)
    while start == end:
        end = random.randint(0, num_nodes - 1)
    
    ques = {
        'u': start, 
        'v': end
    }
    if g.is_directed():
        ques_str = "Calculate the Jaccard coefficient of \
node {} and node {}. For a directed graph, we consider a node's successors as its neighbors.".format(NID(ques['u']), NID(ques['v']))
    else:
        ques_str = "Calculate the Jaccard coefficient of \
node {} and node {}.".format(NID(ques['u']), NID(ques['v']))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of calculating the Jaccard coefficient of node {} and node {}:\n".format(
    #     NID(ques['u']), NID(ques['v']))
    steps_str = "Let's calculate the Jaccard coefficient step by step.\n"
    
    u = ques['u']
    v = ques['v']
    u_nei = list(g.neighbors(u))
    v_nei = list(g.neighbors(v))
    steps_str += "The neighbors of node {}: {}.\n".format(
        NID(u), NID(np.array(u_nei))
    )
    steps_str += "The neighbors of node {}: {}.\n".format(
        NID(v), NID(np.array(v_nei))
    )
    cn = np.array(list(set(u_nei) & set(v_nei)))
    un = np.array(list(set(u_nei) | set(v_nei)))
    # print(cn)
    steps_str += "The common neighbor set of \
node {} and node {} is: {}, and there are {:d} elements.\n".format(
    NID(u), NID(v), NID(cn), len(cn))
    steps_str += "The union neighbor set of \
node {} and node {} is: {}, and there are {:d} elements.\n".format(
    NID(u), NID(v), NID(un), len(un))
    
    n_cn = len(cn)
    n_un = len(un)

    reject = False
    if n_un == 0:
        reject = True
        return steps_str, None, None, reject

    # ans_str = str(ans)
    ans = n_cn/n_un
#     steps_str += "So the Jaccard coefficient is the value of \
# dividing the number of common neighbors by the number of union neighbors, i.e. \
# {:d} / {:d} = {:.4f}.".format(n_cn, n_un, ans)
    steps_str += "So the Jaccard coefficient is the value of \
dividing the number of common neighbors by the number of union neighbors, i.e. \
{:d} / {:d} = ".format(n_cn, n_un)
    
    ans_str = "{:.4f}".format(ans)
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


class Evaluator(FloatEvaluator):

    def __init__(self):
        super().__init__()
