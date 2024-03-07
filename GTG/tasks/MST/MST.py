from GTG.utils.utils import NID, graph_generation_with_edge_weight, make_sample
from GTG.utils.parse_output import parse_list_with_weight, parse_dict_with_weight, parse_edge_list
from GTG.utils.utils import graph_with_egde_weight_to_str, graph_with_egde_weight_to_adj_str
from GTG.utils.evaluation import IntEvaluator

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


TASK_NAME = 'MST'


def question_generation(config, g):
    ques_str = "Output the total weight of the minimum spanning tree (MST) for this graph."

    return ques_str


def answer_and_inference_steps_generation(config, g):
    #在生成graph时将directed设置为0
    if nx.is_connected(g)==False:
        reject = True
        steps_str=None
        ans=None
        ans_str1=None
        ans_str2=None
        return steps_str, ans, ans_str1, ans_str2, reject
    else:
        reject = False
    #当生成的图是非联通图的时候重来
    
    def create_Y(x,v):
        Y = []
        for point in v:
            if point not in x:
                Y.append(point)
        return Y

    def Prim(v, E, x):
 
        weight = 1000
        point1 = 0
        point2 = 0
        # steps_str="Step "+str(len(x))+":\n"
        steps_str="Collected nodes:"
        for i,node in enumerate(x):
            steps_str+=" {}".format(NID(node))
            if i!=len(x)-1:
                steps_str+=","
        steps_str+=". The edge with minimum weight we find: "
        ans_str1=""
        #round：加入edge
        for i in range(len(v)):
            if v[i] in x:  # 确定了顶点集合X中元素在顶点集合v中的下标，间接的确定了在边集合中的一个下标
                for j in range(len(v)):
                    if v[j] in create_Y(x,v):  # 确定了顶点集合Y中元素在集合V中的下标，间接确定了在边集合中的另一个下标
                        if (E[i][j] != -1) and (E[i][j] < weight):  # 循环判断，找到顶点集合X到Y之间最小权值的边
                            weight = E[i][j]
                            point1 = v[i]
                            point2 = v[j]
        ans_str1+="({}, {}, weight:{:d})".format(NID(point1),NID(point2),weight)
        steps_str+="({}, {}, weight:{:d}).\n".format(NID(point1),NID(point2),weight)
        x.append(point2) #在v,x中都是数字
        if len(x)<len(v):
            #steps_str+=","
            ans_str1+=", "
        return steps_str,ans_str1
    
    steps_str = "Let's solve it step by step. We can use the Prim's algorithm. Start from node "
    ans_str1="["
    
    ans=0
    MST = nx.minimum_spanning_tree(g)
    for edge in MST.edges(data=True):
        ans+=edge[2]['weight']
    
        
    adj=np.ones((len(g.nodes),len(g.nodes)),dtype=np.int32)*-1
    for edge in g.edges:
        node1,node2=edge
        adj[node1,node2]=g.edges[node1,node2]['weight']
        adj[node2,node1]=g.edges[node1,node2]['weight']
    
    v=[]   
    x=[0]
    for node in g.nodes:
        adj[node,node]=0
        v.append(node)
    steps_str+="{}.\n".format(NID(0))
    
    count = 0
    while True:
        if count < len(v)-1:
            step,anst=Prim(v, adj, x)
            steps_str+=step
            ans_str1+=anst
            count = count + 1
        else:
            break
    ans_str1+="]"
    steps_str+="These edges make up its minimum spanning tree.\n"
    steps_str+="So the total weight of the minimum spanning tree is "

    ans_str2=str(ans)
    
    return steps_str, ans, ans_str1, ans_str2, reject


def _answer_and_inference_steps_generation(config, g):
    #在生成graph时将directed设置为0
    if nx.is_connected(g)==False:
        reject = True
        steps_str=None
        ans=None
        ans_str1=None
        ans_str2=None
        return steps_str, ans, ans_str1, ans_str2, reject
    else:
        reject = False
    #当生成的图是非联通图的时候重来
    
    def create_Y(x,v):
        Y = []
        for point in v:
            if point not in x:
                Y.append(point)
        return Y

    def Prim(v, E, x):
 
        weight = 1000
        point1 = 0
        point2 = 0
        steps_str="step "+str(len(x))+":\n"
        steps_str+="The nodes that has been collected at this step are"
        for i,node in enumerate(x):
            steps_str+=" {}".format(NID(node))
            if i!=len(x)-1:
                steps_str+=","
        steps_str+=", and the shortest edge we find on the edges connected to the remaining nodes is "
        ans_str1=""
        #round：加入edge
        for i in range(len(v)):
            if v[i] in x:  # 确定了顶点集合X中元素在顶点集合v中的下标，间接的确定了在边集合中的一个下标
                for j in range(len(v)):
                    if v[j] in create_Y(x,v):  # 确定了顶点集合Y中元素在集合V中的下标，间接确定了在边集合中的另一个下标
                        if (E[i][j] != -1) and (E[i][j] < weight):  # 循环判断，找到顶点集合X到Y之间最小权值的边
                            weight = E[i][j]
                            point1 = v[i]
                            point2 = v[j]
        ans_str1+="({}, {}, weight:{:d})".format(NID(point1),NID(point2),weight)
        steps_str+="({}, {}, weight:{:d}).\n".format(NID(point1),NID(point2),weight)
        x.append(point2) #在v,x中都是数字
        if len(x)<len(v):
            #steps_str+=","
            ans_str1+=","
        return steps_str,ans_str1
    
    steps_str = "Starting from any node, find the minimum spanning tree of the graph.\n\
Start from node "
    ans_str1="["
    
    ans=0
    MST = nx.minimum_spanning_tree(g)
    for edge in MST.edges(data=True):
        ans+=edge[2]['weight']
    
        
    adj=np.ones((len(g.nodes),len(g.nodes)),dtype=np.int32)*-1
    for edge in g.edges:
        node1,node2=edge
        adj[node1,node2]=g.edges[node1,node2]['weight']
        adj[node2,node1]=g.edges[node1,node2]['weight']
    
    v=[]   
    x=[0]
    for node in g.nodes:
        adj[node,node]=0
        v.append(node)
    steps_str+="{}, the steps to collect edges are\n".format(NID(0))
    
    count = 0
    while True:
        if count < len(v)-1:
            step,anst=Prim( v, adj, x)
            steps_str+=step
            ans_str1+=anst
            count = count + 1
        else:
            break
    ans_str1+="]"
    steps_str+="These edges make up its minimum spanning tree.\n"
    steps_str+="Thus, the total weight of the minimum spanning tree is "

    ans_str2=str(ans)
    
    return steps_str, ans, ans_str1, ans_str2, reject


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
    g = graph_generation_with_edge_weight(config,False)
    ques_str = question_generation(config, g)
    steps_str, ans, ans_str1, ans_str2, reject = answer_and_inference_steps_generation(config, g)
    
    while reject:
        g = graph_generation_with_edge_weight(config,False)
        ques_str = question_generation(config, g)
        steps_str, ans, ans_str1, ans_str2, reject = answer_and_inference_steps_generation(config, g)
        #answer_tree return 
    
    choi_str, label_str = choices_generation(config, g, ans)

    sample = make_sample(
        TASK_NAME, g, ques_str, ans_str2, steps_str, choi_str, label_str,
        g_str=graph_with_egde_weight_to_str(g),
        g_adj_str=graph_with_egde_weight_to_adj_str(g)
    )
    sample['answer_tree']=ans_str1
    return sample


class Evaluator(IntEvaluator):

    def __init__(self):
        super().__init__()
