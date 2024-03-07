from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import BaseEvaluator
from GTG.utils.parse_output import get_nodes_set, get_nodes_set_from_nodes_str

import networkx as nx
import random
import numpy as np


TASK_NAME = 'page_rank'
DAMPING = 0.85
NUM_ITER = 3


def question_generation(config, g):
    ques = {}
    
    ques_str = "Which node has the largest PageRank value? \
The dampling factor is {:.2f}. \
The number of iterations is {}. The initial PageRank values for all nodes are initialized equally as 1/N, where N is the number of nodes.".format(DAMPING, NUM_ITER)

    return ques, ques_str


def answer_and_inference_steps_generation(config, g, ques):
    # steps_str = "Reasoning process and intermediate results of the PageRank algorithm:\n"
    steps_str = "Let's calculate PageRank step by step.\n"
    steps_str += "All the nodes: {}.\n".format(NID(np.arange(g.number_of_nodes())))

    # set print format of numpy array
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    adj = nx.to_numpy_array(g)
    M = adj / (adj.sum(axis=-1) + 0.01)  # normalized adjacency matrix
    steps_str += "The normalized adjacency matrix M is:\n{}.\n".format(
        M
    )

    num_iterations = NUM_ITER
    N = g.number_of_nodes()
    v = np.ones(N) / N
    d = DAMPING
    M_hat = (d * M + (1 - d) / N)
    
    steps_str += "According to M_hat = (d * M + (1 - d) / N), \
where d is the damping factor {:.2f} and N is the number of nodes, \
the transition probability is:\n{}.\n".format(
            DAMPING, M_hat
        )

    for i in range(num_iterations):
        steps_str += "PageRank values of each node from node {} to node {} at round {} are: {}.\n".format(
            NID(0), NID(N-1), i, v
        )
        v = M_hat @ v
    
    steps_str += "Finally, after {} rounds of iteration, \
the PageRank values of each node from node {} to node {} are: {}.\n".format(
        NUM_ITER, NID(0), NID(N-1), v
    )
    ans_list = np.where(v == np.max(v))[0]  # the node with largest PageRank

    if len(ans_list) > 0.25 * g.number_of_nodes():
        reject = True
    else:
        reject = False

    ans_str_list = ""
    for ans in ans_list:
        ans_str_list += (NID(ans) + " ")
    # steps_str += "Thus, the node with the largest PageRank value is node {}.".format(ans_str)
    steps_str += "So the node with the largest PageRank value is "

    # print(steps_str)
    return steps_str, ans_str_list, ans, reject


def choices_generation(config, g, ques, ans):
    num_choices = 4
    
    false_ans = set()
    while len(false_ans) < num_choices - 1:
        x = random.randint(0, g.number_of_nodes() - 1)
        if x != ans:
            false_ans.add(x)
    
    choi = np.array(list(false_ans) + [ans])
    np.random.shuffle(choi)
    label_str = str(np.arange(num_choices)[choi == ans][0])
    choi_str = "[" + ", ".join([NID(x) for x in choi]) + "]"

    return choi_str, label_str


def generate_a_sample(config):
    g = graph_generation(config)
    ques, ques_str = question_generation(config, g)
    steps_str, ans_str, ans, reject = answer_and_inference_steps_generation(config, g, ques)
    while reject:
        g = graph_generation(config)
        ques, ques_str = question_generation(config, g)
        steps_str, ans_str, ans, reject = answer_and_inference_steps_generation(config, g, ques)

    choi_str, label_str = choices_generation(config, g, ques, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    return sample


class Evaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
    
    def parse_output_ans_str(self, sample, output_ans_str):
        nodes_set = get_nodes_set(sample)
        output_ans = output_ans_str.strip()
        if output_ans in nodes_set:
            return output_ans
        else:
            return None
    
    def check_correctness(self, sample, output_ans):
        ans_set = get_nodes_set_from_nodes_str(str(sample['answer']))
        return output_ans in ans_set
