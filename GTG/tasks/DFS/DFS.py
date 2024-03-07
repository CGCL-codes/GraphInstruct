from GTG.utils.utils import NID, graph_generation, make_sample, edge_list_str_to_graph
from GTG.utils.evaluation import NodeListEvaluator

import networkx as nx
import random
import numpy as np


TASK_NAME = 'DFS'


def question_generation(config, g):
    v = random.randint(0, g.number_of_nodes() - 1)
    ques = {'start': v}
    
    ques_str = "Start from node {}, \
output a sequence of traversal in depth-first search (DFS) order.".format(
    NID(ques['start']))

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of the depth-first search (DFS) algorithm:\n"
    steps_str = "Let's run depth-first search (DFS) step by step.\n"
    
    v = ques['start']
    visited = []
    stack = []
    stack.append(v)
    # steps_str += "Initial state:\n"
    # steps_str += "visited: {}, stack: {}.\n".format(NID(visited), NID(stack))
    while len(stack) != 0:
        v = stack.pop()
        if v not in visited:
            visited.append(v)
            L = []
            for u in g.neighbors(v):
                stack.append(u)
                L.append(u)
            steps_str += "Visit node {}. ".format(NID(v))
            steps_str += "Neighors of node {}: {}.\n".format(NID(v), NID(L))
            # steps_str += "traversal: {}, stack: {}.\n".format(NID(visited), NID(stack))
    
    ans = visited  # the node with largest page rank
    ans_str = NID(visited)
    # steps_str += "The sequence of traversal: {}".format(NID(ans))
    steps_str += "So the DFS traversal is "

    if len(ans) < 5:
        reject = True
    else:
        reject = False

    # print(steps_str)
    return steps_str, ans, ans_str, reject


def _answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of the depth-first search (DFS) algorithm:\n"
    # steps_str = "Let's run DFS step by step.\n"
    steps_str = ""
    
    v = ques['start']
    visited = []
    stack = []
    stack.append(v)
    steps_str += "Initial state:\n"
    steps_str += "visited: {}, stack: {}.\n".format(NID(visited), NID(stack))
    while len(stack) != 0:
        v = stack.pop()
        if v not in visited:
            visited.append(v)
            steps_str += "Visit node {}.\n".format(NID(v))
            for u in g.neighbors(v):
                stack.append(u)

            steps_str += "traversal: {}, stack: {}.\n".format(NID(visited), NID(stack))
    
    ans = visited
    ans_str = NID(visited)
    # steps_str += "The sequence of traversal: {}".format(NID(ans))
    steps_str += "So the sequence of traversal is: "

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

        correct, idx = check_dfs_path(g, path)
        sample['output_path_len'] = len(path)
        sample['output_vailid_path'] = path[:idx]
        sample['output_vailid_path_len'] = idx
        return correct

# [a, c, d, w, x, y, z]
# current: w
# d -> y, z
# c -> x

def check_dfs_path(g, path):
    idx = 0
    visited_nodes = set([path[0]])
    while idx < len(path):
        subpath, valid = get_dfs_subpath(g, visited_nodes, path[idx:])
        idx += len(subpath)
        if not valid:
            return False, idx
        # check subpath contain all the possible nodes
        complete_subpath = dfs(g, subpath[0])
        if len(subpath) != len(complete_subpath):
            return False, idx
        assert set(subpath) == set(complete_subpath)
    assert idx == len(path)
    return True, idx


def get_dfs_subpath(g, visited_nodes, path):
    i = 1
    while i < len(path):
        if g.has_edge(path[i - 1], path[i]):
            if path[i] in visited_nodes:
                return path[:i], False
            else:
                visited_nodes.add(path[i])
        else:
            j = i - 2
            found = False
            while j >= 0:
                # check if previous node has available neighbors
                unvisited_nei = set(g.neighbors(path[j])) - visited_nodes
                if len(unvisited_nei) > 0:
                    if path[i] not in unvisited_nei:
                        return path[:i], False
                    else:
                        found = True
                        visited_nodes.add(path[i])
                        break
                else:
                    j -= 1
            if not found:  # no available next node (no unvisited nodes)
                return path[:i], True
        i += 1
    return path[:i], True


def dfs(g, start):
    v = start
    visited = []
    stack = []
    stack.append(v)
    while len(stack) != 0:
        v = stack.pop()
        if v not in visited:
            visited.append(v)
            for u in g.neighbors(v):
                stack.append(u)
    return visited


# def eval_dfs_list(sample):
#     parse_success = False
#     x = parse_list(sample['output'])
#     if x is not None:
#         parse_success = True
#         dfs_list = x
#         sample['parsed_output']=x
        
#     if parse_success:
#         is_directed=sample['directed']
#         graph_dict = parse_dict(sample['graph_adj'])
#         if judge_dfs(graph_dict,dfs_list,is_directed):
#             sample['correct'] = 1
#         else:
#             sample['correct'] = 0
#     else:
#         sample['parsed_output'] = 'None'
#         sample['correct'] = 0
#     return sample


# def _dfs_judge(adj, dfs, dn, visited):
#     visited[dfs[dn]]=1
#     if (dn==len(adj)-1):
#         return True

#     flag=True
#     for i in range(dn+1,len(adj)):
#         if visited[dfs[i]]==0:
#             if adj[dfs[dn],dfs[i]] ==1:
#                 flag=_dfs_judge(adj,dfs,i,visited)
#             else:
#                 for j in range(0,len(adj)):
#                     if adj[dfs[dn],j]==1 and visited[j]==0:
#                         flag=False
                        
#             if flag==False:
#                 break

#     return flag


# def dfs_judge(adj, dfs, visited):
#     for i in range(0,len(adj)):
#         if visited[dfs[i]]==0:
#             flag=_dfs_judge(adj,dfs,i,visited)
#         if flag==False:
#             break
        
#     return flag


# def judge_dfs(graph_dict, dfs_list, is_directed):

#     G = nx.DiGraph()

#     if is_directed=='True':
#         is_directed=True
#     else:
#         is_directed=False
#     keys=graph_dict.keys()
#     keys_list=list(keys)
#     node_dict={}
#     i=0

#     for key in keys_list:
#         G.add_node(key)
#         node_dict.update({key:i})
#         i+=1

#     for key in keys_list:
#         vlist=graph_dict[key]
#         for v in vlist:
#             if v in G.nodes:
#                 pass
#             else:
#                 G.add_node(v)
#                 node_dict.update({v:i})
#                 i+=1

#     dfs=np.zeros(len(G.nodes),dtype=np.int32)
#     for n,m in enumerate(dfs_list):
#         dfs[n]=node_dict[m]

#     adj=np.zeros((len(G.nodes),len(G.nodes)),dtype=np.int32)
#     for key in keys_list:
#         vlist=graph_dict[key]
#         for v in vlist:
#             if is_directed:
#                 G.add_edge(key,v)
#                 G.add_edge(v,key)
#                 adj[node_dict[key],node_dict[v]]=1
#                 adj[node_dict[v],node_dict[key]]=1
#             else:
#                 G.add_edge(key,v)
#                 adj[node_dict[key],node_dict[v]]=1
            
#     visited=np.zeros(len(G.nodes),dtype=np.int32)

#     flag=dfs_judge(adj,dfs,visited)
    
#     return flag
