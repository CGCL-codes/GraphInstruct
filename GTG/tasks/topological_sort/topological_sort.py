from GTG.utils.utils import NID, generate_random_directed_acyclic_graph, make_sample, edge_list_str_to_graph
from GTG.utils.evaluation import NodeListEvaluator

import networkx as nx
import random
import numpy as np
from collections import deque


TASK_NAME = 'topological_sort'


def dict_formatter(d):
    # key is node id
    l = ['node {}: {}'.format(NID(k), d[k]) for k in d]
    s = '{' + ', '.join(l) + '}'
    return s


def question_generation(config, g):
    ques = {}
    
    ques_str = "Output the topological sorting of this graph. Topological sorting is a linear ordering of the nodes of a directed acyclic graph (DAG) such that for every directed edge, the source node comes before the end nodes in the ordering."

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of the topological sorting algorithm:\n"
    steps_str = "Let's solve it step by step.\n"

    in_degree = {node: 0 for node in g}
    for node in g:
        for neighbor in g.neighbors(node):
            in_degree[neighbor] += 1
    # steps_str += "In-degree of all the nodes: {}.\n".format(
    #     dict_formatter(in_degree)
    # )

    queue = deque([node for node in g if in_degree[node] == 0])
    result = []
    round = 0

    while len(queue):
        # steps_str += "Round {}:\n".format(round)
        round += 1
        steps_str += "Nodes with 0 in-degree: {}.\n".format(NID(list(queue)))
        node = queue.popleft()
        result.append(node)
        steps_str += "Visit node {} and remove it from the graph.\n".format(NID(node))

        for neighbor in g.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        
        # steps_str += "Remove node {} from the graph.\n".format(NID(node))
        # steps_str += "In-degree of all the nodes: {}.\n".format(
        #     dict_formatter(in_degree)
        # )

    # if len(result) != g.number_of_nodes():
    #     has_cycle = True
    # else:
    #     has_cycle = False
    
    ans = result
    ans_str = NID(ans)
    # steps_str += "The result of topological sorting: {}.".format(ans_str)
    steps_str += "So the result of topological sorting is "

    # print(steps_str)
    return steps_str, ans, ans_str


def _answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of the topological sorting algorithm:\n"
    # steps_str = "Let's do it step by step.\n"
    steps_str = ""

    in_degree = {node: 0 for node in g}
    for node in g:
        for neighbor in g.neighbors(node):
            in_degree[neighbor] += 1
    steps_str += "In-degree of all the nodes: {}.\n".format(
        dict_formatter(in_degree)
    )

    queue = deque([node for node in g if in_degree[node] == 0])
    result = []
    round = 0

    while len(queue):
        steps_str += "Round {}:\n".format(round)
        round += 1
        steps_str += "Nodes with 0 in-degree: {}.\n".format(NID(list(queue)))
        node = queue.popleft()
        result.append(node)
        steps_str += "Add node {} to results list.\n".format(NID(node))

        for neighbor in g.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        
        steps_str += "Remove node {} from the graph.\n".format(NID(node))
        steps_str += "In-degree of all the nodes: {}.\n".format(
            dict_formatter(in_degree)
        )

    # if len(result) != g.number_of_nodes():
    #     has_cycle = True
    # else:
    #     has_cycle = False
    
    ans = result
    ans_str = NID(ans)
    # steps_str += "The result of topological sorting: {}.".format(ans_str)
    steps_str += "So the result of topological sorting is "

    # print(steps_str)
    return steps_str, ans, ans_str


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
    g = generate_random_directed_acyclic_graph(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, ques)
    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    return sample


def get_in_degree(adj):
    d = {key:0 for key in adj}
    for u in adj:
        for v in adj[u]:
            d[v] += 1
    return d


def eval_a_sample(sample):
    idx = sample['output'].upper().find('ANSWER')
    if idx != -1:
        sample['output'] = sample['output'][idx + 6:]

    x = parse_list(sample['output'])
    if x is None:
        sample['parsed_output'] = 'None'
        sample['correct'] = 0
        return sample
    else:
        sample['parsed_output'] = str(x)
    
    adj = eval(add_quotation_mark(sample['graph_adj']))
    # check node id:
    if not ((set(x) == set(adj.keys())) & (len(x) == len(adj.keys()))):
        sample['correct'] = 0
        return sample

    in_degree = get_in_degree(adj)
    correct = 1
    queue = deque(x)
    while len(queue):
        u = queue.popleft()
        if in_degree[u] != 0:
            correct = 0
            break
        else:
            nei = adj.pop(u)  # remove 0 in-degree node u
            for v in nei:
                in_degree[v] -= 1  # update in-degree
    
    sample['correct'] = correct
    return sample


class Evaluator(NodeListEvaluator):

    def __init__(self):
        super().__init__()

    def check_correctness(self, sample, output_ans):
        g = edge_list_str_to_graph(sample['graph'], directed=True)
        if len(set(output_ans)) != g.number_of_nodes():
            return False

        correct = True
        queue = deque(output_ans)
        while len(queue):
            u = queue.popleft()
            if g.in_degree(u) != 0:
                correct = False
                break
            else:
                g.remove_node(u)

        return correct
