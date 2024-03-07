import GTG
from GTG.utils.parse_arguments import parse_arguments
from GTG.utils.utils import set_random_seed, display_sample
from GTG.utils.utils import string_to_integer_hashing

from tqdm import tqdm
import pandas as pd
import numpy as np


def main():
    
    config = parse_arguments()
    print("------------------\n{}\n".format(config))
    
    if 'seed' in config:
        set_random_seed(config['seed'])
    else:
        if 'hash_str' in config:
            hash_str = config['hash_str']
        else:
            hash_str = ''
        set_random_seed(
            string_to_integer_hashing(
                config['task'] + hash_str  # map task_name to an int value
            )
        )

    task_module = {
        'shortest_path': GTG.tasks.shortest_path,
        'page_rank': GTG.tasks.page_rank,
        'DFS': GTG.tasks.DFS,
        'BFS': GTG.tasks.BFS,
        'degree': GTG.tasks.degree,
        'common_neighbor': GTG.tasks.common_neighbor,
        'jaccard': GTG.tasks.jaccard,
        'connectivity': GTG.tasks.connectivity,
        'topological_sort': GTG.tasks.topological_sort,
        'cycle': GTG.tasks.cycle,
        'edge': GTG.tasks.edge,
        'neighbor': GTG.tasks.neighbor,
        'predecessor': GTG.tasks.predecessor,
        'bipartite': GTG.tasks.bipartite,
        'ad_location': GTG.tasks.ad_location,
        'fire_station': GTG.tasks.fire_station,

        # JJQ:
        'clustering_coefficient':GTG.tasks.clustering_coefficient,
        'diameter': GTG.tasks.diameter,
        'MST': GTG.tasks.MST,

        # ZCH:
        'euler_path': GTG.tasks.euler_path,
        'hamiltonian_path': GTG.tasks.hamiltonian_path,
        'maximum_flow': GTG.tasks.maximum_flow,
        'connected_component': GTG.tasks.connected_component,
    }[config['task']]

    # display a sample on the screen
    sample = task_module.generate_a_sample(config)
    print(display_sample(sample))

    # save 10 sample examples in a file
    file_example = config['file_output'] + "-example.txt"
    with open(file_example, 'w') as f:
        for i in range(1, 11):
            sample = task_module.generate_a_sample(config)
            f.write(display_sample(sample, i))

    # generate and save all the samples
    df_dict = {key: [] for key in sample.keys()}
    for _ in tqdm(range(config['num_samples'])):
        sample = task_module.generate_a_sample(config)
        for key in df_dict:
            df_dict[key].append(sample[key])
    
    if hasattr(task_module, 'on_finished'):
        task_module.on_finished()

    df = pd.DataFrame.from_dict(df_dict)
    df['id'] = np.arange(len(df))
    df.to_csv(config['file_output'], index=False)

    # save dataset statistics
    ne = df['num_edges'].to_numpy()
    n = df['num_nodes'].to_numpy()
    directed = df['directed'].to_numpy()

    ave_degree = np.empty(len(n), dtype=float)
    ave_degree[directed] = (ne / n)[directed]
    ave_degree[~directed] = (2 * ne / n)[~directed]
    df['ave_degree'] = ave_degree

    df['ave_degree_div_num_nodes'] = ave_degree / n

    is_directed = np.zeros(len(n))
    is_directed[directed] = 1
    df['is_directed'] = is_directed

    a = df.describe()
    a.to_csv(config['file_output'] + "-describe.csv", index=True)


if __name__ == '__main__':

    main()
