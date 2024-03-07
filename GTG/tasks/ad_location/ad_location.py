from GTG.utils.utils import NID, load_ad_graph, make_sample
from GTG.utils.utils import graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str
from GTG.utils.evaluation import eval_node_in_list

import networkx as nx
import random
import numpy as np


TASK_NAME = 'ad_location'
_num_sample = 0
_num_zero = 0


def question_generation(config, g):
    ques = {}
    ques_str = "Given a city network, where the traffic flow is provided, if the advertiser wants to place an advertisement board in the city, which could maximize the number of views, where (which node) should he choose? Please output the answer behind 'Answer:'."

    return ques, ques_str


def answer_and_inference_steps_generation(config, g):
    # steps_str = "Reasoning process and intermediate results of calculating the degree of node {} is:\n".format(
    #     NID(node))
    volume = dict(nx.degree(g, weight='volume'))

    max_value = max(volume.values())
    ans_list = ['<'+str(key)+'>' for key, value in volume.items() if value == max_value]
    
    steps_str = ""

    ans = NID(ans_list)
    ans_str_list = ""
    for ans in ans_list:
        ans_str = NID(ans)
        ans_str_list += (ans_str + " ")
    reject = False
    
    return steps_str, ans, ans_str, reject


def generate_a_sample(config):
    g = load_ad_graph(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)
    
    while reject:
        g = load_ad_graph(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)
   
    choi_str, label_str = "", ""

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str,
        g_str=graph_with_egde_weight_to_str(g, weight_name='volume'),
        g_adj_str=graph_with_egde_weight_to_adj_str(g, weight_name='volume')
    )
    global _num_sample
    _num_sample += 1
    return sample


def eval_a_sample(sample):
    idx = sample['output'].upper().find('ANSWER')
    if idx != -1:
        sample['output'] = sample['output'][idx + 6:]
    sample = eval_node_in_list(sample)
    return sample
