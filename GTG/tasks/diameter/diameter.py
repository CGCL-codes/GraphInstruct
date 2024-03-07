from GTG.utils.utils import NID, graph_generation, make_sample
from GTG.utils.evaluation import IntEvaluator

import networkx as nx
import random
import numpy as np

_num_sample = 0

TASK_NAME = 'diameter'


def question_generation(config, g):
    
    ques_str = "Calculate the diameter of the graph. The diameter is the maximum distance over all pairs of nodes in the graph."

    return ques_str


def answer_and_inference_steps_generation(config, g):
    #在生成graph时将directed设置为0
    if nx.is_connected(g)==False:
        reject = True
        steps_str=None
        ans=None
        ans_str=None
        return steps_str, ans, ans_str, reject
    else:
        reject = False
        ans=nx.diameter(g)
    #当生成的图是非联通图的时候重来
    
    steps_str = "Let's calculate the diameter of the graph step by step.\n"

    #记录每一个节点到其他几点最佳的距离
    ldj=np.zeros((len(g.nodes),len(g.nodes)),dtype=np.int32)
    #max_dict={}
    for i,node1 in enumerate(g.nodes):
        steps_str+="The distance from node {}".format(NID(node1))+" to nodes"
        k=0
        for node2 in g.nodes:
            if node1==node2:
                continue
            k+=1
            if k<len(g.nodes)-1:
                steps_str+=" {},".format(NID(node2))
            else:
                steps_str+=" {}".format(NID(node2))
        steps_str+=" are"
        k=0
        for j,node2 in enumerate(g.nodes):
            if j==i:
                continue
            ldj[i,j]=nx.shortest_path_length(g, source=node1, target=node2)
            k+=1
            if k<len(g.nodes)-1:
                steps_str+=" "+str(ldj[i,j])+","
            else:
                steps_str+=" "+str(ldj[i,j])+", respectively."
        
        maxd=list((np.where(ldj[i]==np.max(ldj[i])))[0])
        maxk=np.max(ldj[i])
        steps_str+=" And the maximum is "+str(maxk)+".\n"
        maxd.append(maxk)
        #max_dict[node1]=maxd
        
    max=nx.diameter(g)
    steps_str+="So the diameter of the graph is "

    ans_str=str(max)
    
    return steps_str, ans, ans_str, reject


def choices_generation(config, g, ans):
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
    g = graph_generation(config,False)
    ques_str = question_generation(config, g)
    steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)
    
    while reject:
        g = graph_generation(config,False)
        ques_str = question_generation(config, g)
        steps_str, ans, ans_str, reject = answer_and_inference_steps_generation(config, g)
   
    choi_str, label_str = choices_generation(config, g, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str, steps_str, choi_str, label_str
    )
    global _num_sample
    _num_sample += 1
    return sample


class Evaluator(IntEvaluator):

    def __init__(self):
        super().__init__()
