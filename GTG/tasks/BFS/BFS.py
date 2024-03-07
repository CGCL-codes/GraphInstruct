from GTG.utils.utils import NID, graph_generation, make_sample, edge_list_str_to_graph
from GTG.utils.evaluation import NodeListEvaluator

import networkx as nx
import random
import numpy as np
from collections import deque


TASK_NAME = 'BFS'


def question_generation(config, g):
    v = random.randint(0, g.number_of_nodes() - 1)
    ques = {'start': v}
    
    ques_str = "Start from node {}, \
output a sequence of traversal in breadth-first search (BFS) order.".format(
    NID(ques['start']))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    steps_str = "Let's run breadth-first search (BFS) step by step.\n"
    
    start = ques['start']
    visited = set()  # Set to keep track of visited nodes
    queue = deque([start])  # Queue for BFS traversal
    visited.add(start)
    traversal = []
    
    # steps_str += "Initial state:\n"
    # steps_str += "visited: {}. queue: {}.\n".format(NID(traversal), NID(list(queue)))

    while len(queue):
        vertex = queue.popleft()
        steps_str += "Visit node {}. ".format(NID(vertex))
        traversal.append(vertex)

        # Traverse adjacent nodes of the current vertex
        unvisited_nei = []
        for neighbor in g.neighbors(vertex):
            if neighbor not in visited:
                queue.append(neighbor)
                unvisited_nei.append(neighbor)
                visited.add(neighbor)
        if len(unvisited_nei) > 0:
            steps_str += "Unvisited neighbors of node {} are {}.\n".format(
                NID(vertex), NID(unvisited_nei)
            )
        else:
            steps_str += "\n"
        # steps_str += "Visit node {}.\n".format(NID(vertex))
        # steps_str += "traversal: {}, queue: {}.\n".format(NID(traversal), NID(list(queue)))

    ans = traversal  # the node with largest page rank
    ans_str = NID(traversal)
    steps_str += "So the BFS traversal is "

    if len(ans) < 5:
        reject = True
    else:
        reject = False

    # print(steps_str)
    return steps_str, ans, ans_str, reject


def _answer_and_inference_steps_generation(config, g, ques):
    steps_str = "Let's run BFS step by step.\n"
    
    start = ques['start']
    visited = set()  # Set to keep track of visited nodes
    queue = deque([start])  # Queue for BFS traversal
    visited.add(start)
    traversal = []
    
    steps_str += "Initial state:\n"
    steps_str += "visited: {}. queue: {}.\n".format(NID(traversal), NID(list(queue)))

    while len(queue):
        vertex = queue.popleft()
        traversal.append(vertex)

        # Traverse adjacent nodes of the current vertex
        for neighbor in g.neighbors(vertex):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    
        steps_str += "Visit node {}.\n".format(NID(vertex))
        steps_str += "traversal: {}, queue: {}.\n".format(NID(traversal), NID(list(queue)))

    ans = traversal  # the node with largest page rank
    ans_str = NID(traversal)
    steps_str += "Thus, the sequence of traversal is "

    if len(ans) < 5:
        reject = True
    else:
        reject = False

    # print(steps_str)
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ques, ans):
    false_ans = []

    # random order
    x = np.array(ans[1:])
    np.random.shuffle(x)
    false_ans.append([ans[0]] + list(x))

    # half random
    cut = len(ans) // 2
    x = np.array(ans[cut:])
    np.random.shuffle(x)
    false_ans.append(ans[:cut] + list(x))

    # cut and concat
    assert g.number_of_nodes() >= 5
    cut = len(ans) // 3
    false_ans.append(ans[:cut] + ans[-cut:] + ans[cut:-cut])

    _choi_list = [ans] + false_ans
    choi_list = []
    ind = np.arange(4)
    np.random.shuffle(ind)
    for i in ind:
        choi_list.append(NID(_choi_list[i]))
    label_str = str(np.arange(4)[ind == 0][0])
    choi_str = "[" + ", ".join(choi_list) + "]"
    
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
    return sample


class Evaluator(NodeListEvaluator):

    def __init__(self):
        super().__init__()

    def check_correctness(self, sample, output_ans):
        g = edge_list_str_to_graph(sample['graph'], directed=sample['directed'])
        path = output_ans
        
        ques = sample['question']
        start_node = ques[len("Start from node "):ques.find(',')]
        if len(path) == 0 or path[0] != start_node:
            sample['output_path_len'] = len(path)
            sample['output_vailid_path'] = 0
            sample['output_vailid_path_len'] = 0
            return False

        nodes_hops = nx.shortest_path_length(g, source=start_node)
        hops_nodes = {}
        for node in nodes_hops:
            hop = nodes_hops[node]
            if hop not in hops_nodes:
                hops_nodes[hop] = [node]
            else:
                hops_nodes[hop].append(node)

        idx = 0
        for k in range(len(hops_nodes)):
            k_hop_nodes = hops_nodes[k]
            _k_hop_nodes = path[idx:idx + len(k_hop_nodes)]
            if set(k_hop_nodes) != set(_k_hop_nodes):
                return False
            idx += len(k_hop_nodes)
        return True


# # wrong, do not use this:
# def is_valid_bfs(graph, bfs):
#     queue = [bfs[0]]    
#     visited = {bfs[0]}  
#     bfs_index = 1

#     while queue:
#         node = queue.pop(0)    
#         node_neighbors = [n for n in graph[node] if n not in visited]

#         for i in range(len(node_neighbors)):
#             if bfs_index + i >= len(bfs) or node_neighbors[i] != bfs[bfs_index + i]:
#                 return False

#         bfs_index += len(node_neighbors) 
#         queue.extend(node_neighbors)   
#         visited.update(node_neighbors) 

#     return bfs_index == len(bfs)
