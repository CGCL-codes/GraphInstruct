import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import hashlib


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_sample(task_name, g, ques_str, ans_str, steps_str=None, choi_str=None, label_str=None,
                g_str=None, g_adj_str=None, g_adj_nl=None):
    sample = {
        'task': task_name,
        'graph': graph_to_edge_list_str(g) if g_str is None else g_str,
        'graph_adj': graph_to_adj_str(g) if g_adj_str is None else g_adj_str,
        'graph_nl': graph_to_natural_language(g) if g_adj_nl is None else g_adj_nl,
        'nodes': NID(np.arange(g.number_of_nodes())),
        'num_nodes': g.number_of_nodes(),
        'num_edges': g.number_of_edges(),
        'directed': g.is_directed(),
        'question': ques_str,
        'answer': ans_str,
    }
    if steps_str is not None:
        sample['steps'] = steps_str
    if choi_str is not None:
        sample['choices'] = choi_str,
    if label_str is not None:
        sample['label'] = label_str

    return sample


def string_to_integer_hashing(s):
    # hash mapping
    digest = hashlib.sha256(s.encode()).hexdigest()
    integer_value = int(digest, 16) % (2**32)
    return integer_value


def display_sample(sample, i=None):
    # return a string which display the sample
    s = ""
    if i is not None:
        s += "####### sample {} #######\n".format(i)
    for key in sample:
        s += "\n>> {}:\n\n{}\n".format(key, sample[key])
    s += "\n"
    return s


def NID(node):
    # return formatted node id str
    formatter = "<{}>".format
    if isinstance(node, np.ndarray) or isinstance(node, list):
        s = '[' + ', '.join([formatter(u) for u in node]) + ']'
    else:
        s = formatter(node)
    return s


def NID_edges(edges):
    s = "["
    s += ", ".join(["({}, {})".format(NID(e[0]), NID(e[1])) for e in edges])
    s += "]"
    return s


def graph_to_edge_list_str(g):
    # edge list in string format: "[(0, 1), (0, 2), ... ]"
    g_str = '['
    for e in g.edges():
        g_str += '(' + NID(e[0]) + ', ' + NID(e[1]) + '), '
    g_str = g_str[:-2] + ']'
    return g_str


def edge_list_str_to_graph(s, directed):
    s = s.split('),')
    for i in range(len(s)):
        x = s[i]
        for r in '[]()':
            x = x.replace(r, '')
        u, v = x.split(',')
        u = u.strip()
        v = v.strip()
        s[i] = (u, v)
    if directed:
        g = nx.DiGraph(s)
    else:
        g = nx.Graph(s)
    return g


def graph_with_egde_weight_to_str(g, weight_name='weight'):
    # edge list in string format: "[(0, 1, weight), (0, 2, weight), ... ]"
    g_str = '['
    for ew in g.edges.data(weight_name):
        g_str += '(' + NID(ew[0]) + ', ' + \
            NID(ew[1]) + ', ' + weight_name +':' + str(ew[2]) + '), '
    g_str = g_str[:-2] + ']'
    return g_str


def graph_to_natural_language(g):
    s = ""
    for u in g.nodes():
        neighbors = list(g.neighbors(u))
        k = len(neighbors)
        if k == 0:
            # do nothing
            pass
        elif k == 1:
            s += 'Node {} is connected to '.format(NID(u))
            s += 'node {}.\n'.format(NID(neighbors[0]))
        else:
            s += 'Node {} is connected to '.format(NID(u))
            s += 'nodes '
            s += ', '.join([NID(v) for v in neighbors])
            s += '.\n'
    s = s.strip()
    return s


def graph_with_edge_weight_to_natural_language(g):
    s = ""
    for u in g.nodes():
        neighbors = list(get_neighbor_and_edge_weight(g, u))
        k = len(neighbors)
        if k == 0:
            # do nothing
            pass
        elif k == 1:
            s += 'Node {} is connected to '.format(NID(u))
            s += 'node {} (weight: {}).\n'.format(
                NID(neighbors[0][0]), neighbors[0][1]
            )
        else:
            s += 'Node {} is connected to '.format(NID(u))
            s += 'nodes '
            s += ', '.join([NID(v[0]) + ' (weight: {})'.format(v[1]) for v in neighbors])
            s += '.\n'
    return s


def graph_to_adj_str(g):
    s = "{"
    for u in g.nodes():
        s += '{}: ['.format(NID(u))
        for v in g.neighbors(u):
            s += NID(v) + ', '
        if s[-1] == '[':
            s += '],\n'
        else:
            s = s[:-2] + '],\n'
    s = s[:-2] + '}'
    return s


def graph_with_egde_weight_to_adj_str(g, weight_name='weight'):
    s = "{"
    for u in g.nodes():
        s += '{}: ['.format(NID(u))
        for v, w in get_neighbor_and_edge_weight(g, u, weight_name):
            s += '(' + NID(v) + ', '+ weight_name +':{}), '.format(w)
        if s[-1] == '[':
            s += '],\n'
        else:
            s = s[:-2] + '],\n'
    s = s[:-2] + '}'
    return s


def has_completely_isolated_node(g):
    # out-degree + in-degree for directed graphs
    flag = any(degree == 0 for node, degree in g.degree())
    return flag


def id_random_re_mapping(g, return_mapping=False):
    edges = np.array(g.edges())
    num_nodes = g.number_of_nodes()
    all_id = np.arange(num_nodes)
    np.random.shuffle(all_id)

    def mapping(u):
        return all_id[u]

    np.vectorize(mapping)
    new_edges = mapping(edges)

    if g.is_directed():
        new_g = nx.DiGraph(list(new_edges))
    else:
        new_g = nx.Graph(list(new_edges))

    if return_mapping:
        return new_g, mapping
    else:
        return new_g


def shuffle_edges(g):
    edges = np.array(g.edges())
    np.random.shuffle(edges)
    edges = list(edges)
    if g.is_directed():
        new_g = nx.DiGraph(edges)
    else:
        new_g = nx.Graph(edges)
    return new_g


def _randon_sample_ave_degree(num_nodes):
    assert num_nodes >= 4
    ave_degree = random.randint(2, min(int(np.ceil(num_nodes/3)), 10))
    return ave_degree


def generate_random_graph(num_nodes, ave_degree=None, directed=None):
    # print(type(num_nodes))
    if not isinstance(num_nodes, int):
        # uniformly sample num_nodes in range [a, b]
        num_nodes = random.randint(num_nodes[0], num_nodes[1])  ###

    if ave_degree is None:
        # sample average degree
        ave_degree = _randon_sample_ave_degree(num_nodes)

    # directed / undirected
    if directed is None:
        directed = (random.random() < 0.5)  # 50% directed

    def get_graph():
        if directed:
            link_prob = ave_degree / (num_nodes - 1)  ###
        else:
            link_prob = 2 * ave_degree / (num_nodes - 1)  ###
        return nx.fast_gnp_random_graph(
            n=num_nodes, p=link_prob,
            directed=directed
        )

    g = get_graph()
    while has_completely_isolated_node(g):
        g = get_graph()

    # randomly re-map node id
    g = id_random_re_mapping(g)

    return g


def generate_barabasi_albert_graph(num_nodes, ave_degree=None):
    if not isinstance(num_nodes, int):
        # uniformly sample num_nodes in range [a, b]
        num_nodes = random.randint(num_nodes[0], num_nodes[1])  ###

    if ave_degree is None:
        # sample average degree
        ave_degree = _randon_sample_ave_degree(num_nodes)

    n = num_nodes
    # ave_degree = 2 * (n - m) * m / n
    # assume:
    m = int((ave_degree * n / 2) / (n - ave_degree))

    def get_graph():
        return nx.barabasi_albert_graph(n=num_nodes, m=m)

    g = get_graph()
    while has_completely_isolated_node(g):
        g = get_graph()

    # shuffle edges (shuffle neighbors)
    g = shuffle_edges(g)
    # randomly re-map node id
    g = id_random_re_mapping(g)

    return g


def generate_watts_strogatz_graph(num_nodes, ave_degree=None):
    if not isinstance(num_nodes, int):
        # uniformly sample num_nodes in range [a, b]
        num_nodes = random.randint(num_nodes[0], num_nodes[1])  ###

    if ave_degree is None:
        # sample average degree
        ave_degree = _randon_sample_ave_degree(num_nodes)

    p = random.uniform(0.15, 0.45)

    def get_graph():
        return nx.watts_strogatz_graph(n=num_nodes, k=ave_degree, p=p)

    g = get_graph()
    while has_completely_isolated_node(g):
        g = get_graph()

    # shuffle edges (shuffle neighbors)
    g = shuffle_edges(g)
    # randomly re-map node id
    g = id_random_re_mapping(g)

    return g


def graph_generation(config, directed=None):
    num_nodes_range = eval(config['num_nodes_range'])

    def _generate():
        if directed is None:
            p = random.uniform(0, 1)
            if p < 0.5:
                g = generate_random_graph(num_nodes_range)
            elif p < 0.75:
                g = generate_barabasi_albert_graph(num_nodes_range)
            else:
                g = generate_watts_strogatz_graph(num_nodes_range)
        elif directed:
            g = generate_random_graph(num_nodes_range, directed=True)
        else:
            p = random.uniform(0, 1)
            if p < 0.5:
                g = generate_random_graph(num_nodes_range, directed=False)
            elif p < 0.75:
                g = generate_barabasi_albert_graph(num_nodes_range)
            else:
                g = generate_watts_strogatz_graph(num_nodes_range)
        return g
    
    while True:
        g = _generate()
        if g.is_directed():
            ne = g.number_of_edges()
        else:
            ne = 2 * g.number_of_edges()
        ave_degree = ne / g.number_of_nodes()
        if ave_degree > 0.75 * g.number_of_nodes():
            continue
        else:
            break

    return g


def graph_generation_with_edge_weight(config, directed=None):
    _g = graph_generation(config, directed)
    
    if _g.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    
    for e in _g.edges:
        ew = random.randint(1, 10)  # return random integer in range [a, b]
        g.add_edge(e[0], e[1], weight=ew)
        g.edges[e]['weight'] = ew

    return g


def generate_random_directed_acyclic_graph(config):
    """Generate a random DAG with the given number of nodes and edge probability."""

    num_nodes = random.randint(
        *eval(config['num_nodes_range'])
    )
    ave_degree = _randon_sample_ave_degree(num_nodes)
    link_prob = ave_degree / (num_nodes - 1)

    def get_graph():
        g = nx.DiGraph()
        nodes = np.arange(num_nodes)
        for u in nodes:
            g.add_node(u)
        np.random.shuffle(nodes)
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if random.random() < link_prob:
                    g.add_edge(i, j)
                    if nx.is_directed_acyclic_graph(g):
                        continue
                    else:
                        g.remove_edge(i, j)
        assert nx.is_directed_acyclic_graph(g)
        return g
    
    g = get_graph()
    while has_completely_isolated_node(g):
        g = get_graph()

    # shuffle edges (shuffle neighbors)
    g = shuffle_edges(g)
    # randomly re-map node id
    g = id_random_re_mapping(g)

    return g


def generate_bipartite_graph(config):
    num_nodes = random.randint(
        *eval(config['num_nodes_range'])
    )

    def get_graph():
        # seperate node_set_1 and node_set_2
        n1 = int(random.uniform(0.25, 0.5) * num_nodes)
        n2 = num_nodes - n1
        link_prob = random.uniform(0.2, 0.4)

        g = nx.Graph()
        g.add_nodes_from(range(n1), bipartite=0)
        g.add_nodes_from(range(n1, n1 + n2), bipartite=1)

        # Add random edges between the bipartite sets
        for u in range(n1):
            for v in range(n1, n1 + n2):
                if random.random() < link_prob:  # Add an edge with 50% probability
                    g.add_edge(u, v)
        return g, n1, n2

    g, n1, n2 = get_graph()
    while has_completely_isolated_node(g) or not nx.is_connected(g):
        g, n1, n2 = get_graph()

    return g, n1, n2


def generate_Hamiltonian_path_graph(config, directed=None):
    num_nodes = random.randint(
        *eval(config['num_nodes_range'])
    )
    path = np.arange(num_nodes)
    np.random.shuffle(path)  # generate a valid path

    if directed is None:
        directed = (random.random() < 0.5)  # 50% directed
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    # add edges of the Hamiltonian path
    for i in range(num_nodes - 1):
        g.add_edge(path[i], path[i + 1])

    # add some random edges
    ave_degree = _randon_sample_ave_degree(num_nodes) - 1
    link_prob = ave_degree / (num_nodes - 1)

    nodes = np.arange(num_nodes)
    np.random.shuffle(nodes)
    for idx, i in enumerate(nodes):
        if directed:
            dst = nodes
        else:
            dst = nodes[idx + 1:]
        for j in dst:
            if i == j:
                continue
            if random.random() < link_prob:
                g.add_edge(i, j)
    
    # shuffle edges (shuffle neighbors)
    g = shuffle_edges(g)
    # randomly re-map node id
    g, mapping = id_random_re_mapping(g, return_mapping=True)
    path = mapping(path)

    return g, path


def generate_Euler_path_graph(config, directed=None):
    num_nodes = random.randint(
        *eval(config['num_nodes_range'])
    )
    ave_degree = _randon_sample_ave_degree(num_nodes)

    if directed is None:
        directed = (random.random() < 0.5)  # 50% directed
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    for u in range(num_nodes):
        g.add_node(u)

    def sample_node():
        return random.randint(0, num_nodes - 1)

    num_edges = num_nodes * ave_degree
    edge_set = set()
    path = [sample_node()]

    def next_node_ok(v):
        current = path[-1]
        flag = True
        if (current, v) in edge_set:
            flag = False
        elif v == current:
            flag = False
        elif not g.is_directed() and g.degree(v) >= g.number_of_nodes() - 1:
            flag = False
        elif g.is_directed() and g.out_degree(v) >= g.number_of_nodes() - 1:
            flag = False
        return flag
    
    def add_next_to_graph(v):
        current = path[-1]
        path.append(v)
        g.add_edge(current, v)
        edge_set.add((current, v))
        if not g.is_directed():
            edge_set.add((v, current))

    # generate Euler path and graph
    for _ in range(num_edges):
        all_nodes = np.arange(num_nodes)
        np.random.shuffle(all_nodes)
        found = False
        for v in all_nodes:
            if next_node_ok(v):
                add_next_to_graph(v)
                found = True
                break
        if not found:
            break

    # add zero-degree nodes
    for v in range(num_nodes):
        if g.degree(v) == 0:
            add_next_to_graph(v)

    # shuffle edges (shuffle neighbors)
    g = shuffle_edges(g)
    # randomly re-map node id
    g, mapping = id_random_re_mapping(g, return_mapping=True)
    path = np.array(path)
    path = mapping(path)

    return g, path


def load_ad_graph(config, radius=3, size_limit=20):
    netfile = "/home/songxiran/GraphLLM_code_and_data/data/dataset/stage3/ChicagoRegional_flow.tntp"
    net = pd.read_csv(netfile, skiprows=8, sep='\t', engine='python', header=0)
    G = nx.DiGraph()
    for index, row in net.iterrows():
        head = row['Head ']
        tail = row['Tail ']
        volume = row['Volume ']
        cost = row['Cost ']
        G.add_edge(head, tail, volume=round(volume, 2), cost=round(cost, 2))
    while True:
        # np.random.seed(datetime.datetime.now())
        nid = np.random.randint(G.number_of_nodes())
        if nid in G:
            g = nx.ego_graph(G, nid, radius=radius, undirected=True)
            if g.number_of_nodes() < size_limit:
                # print(nid)
                break
        else:
            continue
    l1 = sorted(list(map(int,g.nodes())))#原始节点 排序
    l2 = range(g.number_of_nodes())#新节点 排序
    ##新老节点一一对应
    nodes = dict(map(lambda x,y:[x,y],l1,l2)) 
    new_g=nx.DiGraph()
    for e in g.edges():
        new_g.add_edge(nodes[e[0]], nodes[e[1]], volume=g.edges[e]['volume'], cost=g.edges[e]['cost'])

    return new_g



def load_fire_graph(config, radius=3, size_limit=20):
    netfile = "/home/songxiran/GraphLLM_code_and_data/data/dataset/stage3/ChicagoRegional_net.tntp"
    net = pd.read_csv(netfile, skiprows=8, sep='\t', engine='python', header=0)
    G = nx.DiGraph()
    for index, row in net.iterrows():
        head = row['init_node']
        tail = row['term_node']
        length = row['length']
        G.add_edge(head, tail, length=round(length, 2))
    while True:
        # np.random.seed(datetime.datetime.now())
        nid = np.random.randint(G.number_of_nodes())
        if nid in G:
            g = nx.ego_graph(G, nid, radius=radius, undirected=True)
            if g.number_of_nodes() < size_limit:
                # print(nid)
                break
        else:
            continue
    l1 = sorted(list(map(int,g.nodes())))#原始节点 排序
    l2 = range(g.number_of_nodes())#新节点 排序
    ##新老节点一一对应
    nodes = dict(map(lambda x,y:[x,y],l1,l2)) 
    new_g=nx.DiGraph()
    for e in g.edges():
        new_g.add_edge(nodes[e[0]], nodes[e[1]], length=g.edges[e]['length'])

    return new_g


def get_neighbor_and_edge_weight(g, u, weight_name='weight'):
    return [(item[0], item[1][weight_name]) for item in g[u].items()]
