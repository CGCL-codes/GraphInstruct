from GTG.utils.utils import NID, load_city_graph, make_sample
from GTG.utils.utils import graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str
from GTG.utils.evaluation import eval_node_in_list

import networkx as nx
import random
import numpy as np


TASK_NAME = 'blackhole'
_num_sample = 0
_num_zero = 0


def question_generation(config, g):
    ques = {}
    ques_str = "The phenomenon of a city black hole refers to a place where the incoming traffic volume exceeds the outgoing traffic volume. Please identify the most severe place of city black hole phenomena in the given city network graph."

    return ques, ques_str


def answer_and_inference_steps_generation(config, g):
    total_volume = []
    for i in range(g.number_of_nodes()):
        in_volume = g.in_degree(i, weight='volume')
        out_volume = g.out_degree(i, weight='volume')
        total_volume.append(in_volume-out_volume)
    total_volume = np.array(total_volume)
    ans_list = np.where(total_volume == max(total_volume))[0]

    ans_str_list = ""
    for ans in ans_list:
        ans_str = NID(ans)
        ans_str_list += (ans_str + " ")
    steps_str = ""

    ans = NID(ans_list)
    reject = False
    return steps_str, ans, ans_str, reject


def generate_a_sample(config):
    g = load_city_graph(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)
    
    while reject:
        g = load_city_graph(config)
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
