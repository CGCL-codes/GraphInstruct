from GTG.utils.utils import NID, load_fire_graph, make_sample
from GTG.utils.utils import graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str
from GTG.utils.evaluation import eval_a_node

import networkx as nx
import random
import numpy as np


TASK_NAME = 'fire_station'
_num_sample = 0
_num_zero = 0


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    while True:
        event_node = random.randint(0, num_nodes - 1)    # the event location
        fire_station_nodes = np.random.choice(num_nodes-1, size=3, replace=False)
        if event_node not in fire_station_nodes:
            flag = False
            for node in fire_station_nodes:
                if not nx.has_path(g, event_node, node):
                    flag = True
            if not flag:
                break
    ques = {
        'u': event_node,
        'v': fire_station_nodes
    }
    ques_str = "Given a city network, where the traffic distance is provided. Currently, a fire has occurred at location {}, and as an emergency response system for fire incidents, you need to notify the nearest fire station to proceed to the scene. It is known that there are fire stations at locations {}, {}, and {} in the vicinity. Following the principle of prioritizing the closest distance, which fire station (node) should you notify? Please output the answer behind 'Answer:'.".format(NID(event_node), NID(fire_station_nodes[0]), NID(fire_station_nodes[1]), NID(fire_station_nodes[2]))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    event_node = ques['u']
    distance = []
    for fire_station in ques['v']:
        distance.append(nx.shortest_path_length(g, source=event_node, target=fire_station))
    min_index = np.argmin(np.array(distance))
    steps_str = ""

    ans = ques['v'][min_index]
    ans_str = str(ans)
    reject = False
    return steps_str, ans, ans_str, reject


def generate_a_sample(config):
    g = load_fire_graph(config, radius=4, size_limit=40)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g, ques)
    
    while reject:
        g = load_fire_graph(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)
   
    choi_str, label_str = "", ""

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str,
        g_str=graph_with_egde_weight_to_str(g, weight_name='length'),
        g_adj_str=graph_with_egde_weight_to_adj_str(g, weight_name='length')
    )
    global _num_sample
    _num_sample += 1
    return sample


def eval_a_sample(sample):
    idx = sample['output'].upper().find('ANSWER')
    if idx != -1:
        sample['output'] = sample['output'][idx + 6:]
    sample = eval_a_node(sample)
    return sample
