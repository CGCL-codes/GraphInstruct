from GTG.utils.utils import NID, graph_generation_with_edge_weight, graph_with_edge_weight_to_natural_language, make_sample, graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str
from GTG.utils.evaluation import IntEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'maximum_flow'


def question_generation(config, start_node, end_node):
    ques_str = "Calculate the maximum flow between node <{}> and node <{}> in this graph. Given a directed graph with capacities assigned to its edges, the maximum flow from a source node to a sink node is the maximum amount of flow that can be sent from the source to the sink, respecting the capacity constraints on each edge. The goal is to find the optimal way to route flow through the network to maximize the flow from source to sink.".format(start_node, end_node)
    return ques_str


def generate_maximum_flow_graph(config):
    g = graph_generation_with_edge_weight(config, directed=True)
    start_node = random.choice(list(g.nodes))
    end_node = random.choice(list(g.nodes))
    while start_node == end_node:
        end_node = random.choice(list(g.nodes))

    return g, start_node, end_node  


def answer_and_inference_steps_generation(config, g, start_node, end_node):
    ans, step_str = edmonds_karp(g, start_node, end_node)
    # ans_str = f"The maximum flow from node <{start_node}> to node <{end_node}> is {ans}."
    ans_str = str(ans)

    return step_str, ans, ans_str


def edmonds_karp(graph, start_node, end_node):
    flow = 0
    paths_with_flow = []
    step_str = "Let's solve it step by step. We can use the Edmonds-Karp algorithm. The paths with the corresponding capacity are as follows:\n"
    while True:
        # 使用 BFS 找到增广路径
        queue = [start_node]
        paths = {start_node: []}

        while queue:
            node = queue.pop(0)
            for neighbor in graph.neighbors(node):
                residual = graph.edges[node, neighbor]['weight'] - graph.edges[node, neighbor].get('flow', 0)
                if residual > 0 and neighbor not in paths:
                    paths[neighbor] = paths[node] + [(node, neighbor)]
                    if neighbor == end_node:
                        break
                    queue.append(neighbor)

        # 如果没有找到增广路径，跳出循环
        if end_node not in paths:
            break

        # 计算增广路径上的最小残余容量
        min_residual = min(graph.edges[u, v]['weight'] - graph.edges[u, v].get('flow', 0) for u, v in paths[end_node])
        step_str += f"Found augmenting path: [{', '.join(f'<{str(u)}>' for u, v in paths[end_node])}, <{str(end_node)}>] with minimum residual capacity of {min_residual}."
        # 更新增广路径上的流量
        for u, v in paths[end_node]:
            graph.edges[u, v]['flow'] = graph.edges[u, v].get('flow', 0) + min_residual
            if graph.has_edge(v, u):
                graph.edges[v, u]['flow'] = graph.edges[v, u].get('flow', 0) - min_residual
            else:
                graph.add_edge(v, u, flow=-min_residual, weight=0)

        # 添加路径和流量到列表
        paths_with_flow.append((paths[end_node], min_residual))
        step_str += f" Updated the flow along this path. Total flow: {flow}+{min_residual}={flow+min_residual};\n"
        # 更新总流量
        flow += min_residual
    step_str += f"Thus, the maximum flow from node <{start_node}> to node <{end_node}> is "

    return flow, step_str


def choices_generation(config, ans):
    num_choices = 4
    false_ans = set()
    if ans != 0:
        false_ans.add(0)
    while len(false_ans) < num_choices - 1 :
        x = random.randint(1,10)
        if x+ans not in false_ans:
            false_ans.add(x+ans)
    
    choi = np.array(list(false_ans) + [ans])
    np.random.shuffle(choi)
    label_str = str(np.arange(num_choices)[choi == ans][0])
    choi_str = "[" + ", ".join(["{}".format(x) for x in choi]) + "]"

    return choi_str, label_str


def generate_a_sample(config):
    g, start_node, end_node = generate_maximum_flow_graph(config)
    ques_str = question_generation(config, start_node, end_node)
    steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, start_node, end_node)

    while ans == 0:
        g, start_node, end_node = generate_maximum_flow_graph(config)
        ques_str = question_generation(config, start_node, end_node)
        steps_str, ans, ans_str = answer_and_inference_steps_generation(config, g, start_node, end_node)

    choi_str, label_str = choices_generation(config, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str,
        g_str = graph_with_egde_weight_to_str(g),
        g_adj_str = graph_with_egde_weight_to_adj_str(g),
        g_adj_nl=graph_with_edge_weight_to_natural_language(g)
    )
    return sample


class Evaluator(IntEvaluator):

    def __init__(self):
        super().__init__()
