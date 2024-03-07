from GTG.utils.utils import NID, make_sample, graph_generation, generate_random_graph
from GTG.utils.evaluation import NodeSetEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'connected_component'


def question_generation(config, g):
    num_nodes = g.number_of_nodes()
    start_node = random.randint(0, num_nodes - 1)    # return random integer in range [a, b]
    is_directed = g.is_directed()
    if is_directed:
        ques_str = "Find the strongly connected component \
containing node {}. ".format(NID(start_node))
    else:
        ques_str = "Find the connected component \
containing node {}. ".format(NID(start_node))
    return ques_str, start_node


def answer_and_inference_steps_generation(g, start_node):
    ans, step_str = find_connected_component(g, start_node)
    is_directed = g.is_directed()

    ans_node_set = set(ans)
    if is_directed:
        # check correctness
        for nx_node_set in nx.strongly_connected_components(g):
            if start_node in nx_node_set:
                assert ans_node_set == nx_node_set
                break
        ans_str = NID(ans)
    else:
        for nx_node_set in nx.connected_components(g):
            if start_node in nx_node_set:
                assert ans_node_set == nx_node_set
                break
        ans_str = NID(ans)

    return step_str, ans, ans_str


def tarjan(graph_dict, start_node):
    stack = []
    low = {}
    dfs = {}
    on_stack = {}
    result = []
    index = [0]
    step_str = "Let's find the strongly connected component step by step, and we can use the Tarjan algorithm.\n"
    # step_str += "In Tarjan algorithm, we utilize two elements: DFS number and Low link value.\n"
    # step_str += "The DFS number is assigned to each node according to the sequence in which they are visited in the Depth-First Search. It represents the 'timestamp' of when the node is discovered during the search process.\n"
    # step_str += "The Low link value is the smallest DFS number of any node reachable from the original node, including the node itself. It helps in identifying the root nodes of strongly connected components.\n"
    # step_str += "With these elements defined, let's proceed to find the strongly connected component in the graph.\n"

    def strongconnect(node, step_str):
        low[node] = dfs[node] = index[0]
        index[0] += 1
        stack.append(node)
        on_stack[node] = True
        step_str += "Search from node {}.\n".format(
            NID(node), NID(node), low[node]
        )

        for neighbor in graph_dict[node]:
            if neighbor not in dfs:
                step_str += "Visit neighbor node {}.\n".format(
                    NID(neighbor), NID(node)
                )
                _, step_str = strongconnect(neighbor, step_str)
                low[node] = min(low[node], low[neighbor])
            elif on_stack[neighbor]:
                low[node] = min(low[node], dfs[neighbor])

        if low[node] == dfs[node]:
            connected_component = []   # NID
            while True:
                w = stack.pop()
                on_stack[w] = False
                connected_component.append(w)
                if w == node:
                    break
            result.append(connected_component)
            step_str += "Node {} is a root node, find a new strongly connected component: {}.\n".format(
                NID(node), NID(connected_component)
            )

        return node, step_str

    _, step_str = strongconnect(start_node, step_str)
    # step_str += f"Finished DFS at start node <{start_node}>, all strongly connected components found until now are {result}.\n"

    for component in result:
        if start_node in component:
            step_str += "So the strongly connected component containing \
node {} is ".format(NID(start_node))
            return component, step_str

    return [], step_str


def _tarjan(graph_dict, start_node):
    stack = []
    low = {}
    dfs = {}
    on_stack = {}
    result = []
    index = [0]
    step_str = "Let's find the strongly connected component step by step, and we can use the Tarjan algorithm.\n"
    step_str += "In Tarjan algorithm, we utilize two elements: DFS number and Low link value.\n"
    step_str += "The DFS number is assigned to each node according to the sequence in which they are visited in the Depth-First Search. It represents the 'timestamp' of when the node is discovered during the search process.\n"
    step_str += "The Low link value is the smallest DFS number of any node reachable from the original node, including the node itself. It helps in identifying the root nodes of strongly connected components.\n"
    step_str += "With these elements defined, let's proceed to find the strongly connected component in the graph.\n"

    def strongconnect(node, step_str):
        low[node] = dfs[node] = index[0]
        index[0] += 1
        stack.append(node)
        on_stack[node] = True
        step_str += "Start Depth-First Search at node {}. Set DFS number and Low link value of node {} to {} and add it to the stack.\n".format(
            NID(node), NID(node), low[node]
        )

        for neighbor in graph_dict[node]:
            if neighbor not in dfs:
                step_str += "Explore neighbor node {} of node {}.\n".format(
                    NID(neighbor), NID(node)
                )
                _, step_str = strongconnect(neighbor, step_str)
                low[node] = min(low[node], low[neighbor])
            elif on_stack[neighbor]:
                low[node] = min(low[node], dfs[neighbor])

        if low[node] == dfs[node]:
            connected_component = []   # NID
            while True:
                w = stack.pop()
                on_stack[w] = False
                connected_component.append(w)
                if w == node:
                    break
            result.append(connected_component)
            step_str += "Node {} is a root node, \
pop nodes from stack to form a new strongly \
connected component: {}\n".format(
    NID(node), NID(connected_component)
)

        return node, step_str

    _, step_str = strongconnect(start_node, step_str)
    # step_str += f"Finished DFS at start node <{start_node}>, all strongly connected components found until now are {result}.\n"

    for component in result:
        if start_node in component:
            step_str += "Thus, strongly connected component containing \
node {} is ".format(NID(start_node))
            return component, step_str

    return [], step_str


def find_connected_component(g, start_node):
    # 将NetworkX图转换为字典形式的图
    graph_dict = {node: set(neighbors) for node, neighbors in g.adjacency()}

    # 判断图是有向图还是无向图
    is_directed = g.is_directed()

    if is_directed:
        # 对于有向图，使用Tarjan的算法找到强连通分量
        component, step_str = tarjan(graph_dict, start_node)
        return component, step_str
    else:
        # 对于无向图，使用深度优先搜索找到连通分量
        visited = set()
        component = []
        stack = [start_node]
        step_str = "Let's find the connected component step by step, and we can use Depth-First Search.\n"
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.append(node)
                stack.extend(graph_dict[node] - visited)
                step_str += "Visit node {}, add it to the connected component. \
The current connected component is {}.\n".format(NID(node), NID(component))
        
        step_str += "Thus, the connected component \
containing node {} is ".format(NID(start_node))
        return component, step_str


cc_node_ratio_list = []


def generate_a_sample(config):

    def generate():
        # g = None
        # # TODO 2: reject only 1 node
        # while True:
        #     g = graph_generation(config)
        #     # 检查连通分量和强连通分量的大小
        #     if nx.is_directed(g):
        #         components = nx.strongly_connected_components(g)
        #     else:
        #         components = nx.connected_components(g)
        #     if all(len(c) > 1 for c in components):
        #         # 所有连通分量和强连通分量都包含多于一个节点，跳出循环
        #         break
        g = graph_generation(config)
        # num_nodes_range = eval(config['num_nodes_range'])
        # g = generate_random_graph(num_nodes_range)
        ques_str, start_node = question_generation(config, g)
        steps_str, ans, ans_str = answer_and_inference_steps_generation(g, start_node)
        cc_node_ratio = len(ans) / g.number_of_nodes()
        return cc_node_ratio, g, ques_str, start_node, steps_str, ans, ans_str
        
    def check_sample_ok(cc_node_ratio):
        # # TODO 3: Ensure 25%, 50%, 75% of cc_node_ratio is not 1
        temp_list = cc_node_ratio_list + [cc_node_ratio]
        # for p in [25, 50, 75]:
        #     if len(temp_list) > 3 and np.percentile(temp_list, p) == 1:
        #         return False
        if np.percentile(temp_list, 90) > 0.90:
            return False
        return True

    cc_node_ratio, g, ques_str, start_node, steps_str, ans, ans_str = generate()
    while not check_sample_ok(cc_node_ratio):
        cc_node_ratio, g, ques_str, start_node, steps_str, ans, ans_str = generate()
    
    cc_node_ratio_list.append(cc_node_ratio)
    
    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str,
    )
    sample['cc_node_ratio'] = cc_node_ratio
    return sample


def on_finished():
    for p in [2, 5, 10, 25, 50, 75, 90, 95]:
        print("### cc_node_ratio_list percentile {}%: {}".format(
            p, np.percentile(cc_node_ratio_list, p)
        ))


class Evaluator(NodeSetEvaluator):

    def __init__(self):
        super().__init__()
