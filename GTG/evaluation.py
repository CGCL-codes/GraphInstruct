import GTG
from GTG.utils.parse_arguments import parse_arguments
from GTG.utils.utils import set_random_seed

from tqdm import tqdm
import pandas as pd


def main():
    
    config = parse_arguments()
    print("------------------\n{}\n".format(config))

    evaluator = {
        'topological_sort': GTG.tasks.topological_sort.Evaluator,
        'hamiltonian_path': GTG.tasks.hamiltonian_path.Evaluator,
        'euler_path': GTG.tasks.euler_path.Evaluator,
        'DFS': GTG.tasks.DFS.Evaluator,
        'BFS': GTG.tasks.BFS.Evaluator,
        'bipartite': GTG.tasks.bipartite.Evaluator,
        'shortest_path': GTG.tasks.shortest_path.Evaluator,  # int
        'page_rank': GTG.tasks.page_rank.Evaluator,
        'maximum_flow': GTG.tasks.maximum_flow.Evaluator,  # int
        'degree': GTG.tasks.degree.Evaluator,  # int
        'diameter': GTG.tasks.diameter.Evaluator,  # int
        'common_neighbor': GTG.tasks.common_neighbor.Evaluator,  # int
        'jaccard': GTG.tasks.jaccard.Evaluator,  # float
        'connectivity': GTG.tasks.connectivity.Evaluator,  # bool
        'cycle': GTG.tasks.cycle.Evaluator,  # bool
        'edge': GTG.tasks.edge.Evaluator,  # bool
        'neighbor': GTG.tasks.neighbor.Evaluator,
        'predecessor': GTG.tasks.predecessor.Evaluator,
        'MST': GTG.tasks.MST.Evaluator,  # int
        'clustering_coefficient': GTG.tasks.clustering_coefficient.Evaluator,  # float
        'connected_component': GTG.tasks.connected_component.Evaluator,
    }[config['task']]()

    df_ans = pd.read_csv(config['file_answer'], dtype = {'id': int, 'output': str})
    df_data = pd.read_csv(config['file_dataset'])
    df = df_data.merge(df_ans, left_on='id', right_on='id')

    df_result_dict = {key: [] for key in df.keys()}
    df_result_dict['output_ans_str'] = []
    df_result_dict['output_ans'] = []
    df_result_dict['correct'] = []

    if config['task'] == 'DFS':
        df_result_dict['output_path_len'] = []
        df_result_dict['output_vailid_path_len'] = []
        df_result_dict['output_vailid_path'] = []

    for i, sample in tqdm(df.iterrows()):
        sample = dict(sample)
        evaluator.eval_a_sample(sample)
        for key in df_result_dict:
            if key not in sample:
                df_result_dict[key].append(None)
            else:
                df_result_dict[key].append(sample[key])

    df_result = pd.DataFrame.from_dict(df_result_dict)
    accuracy = df_result['correct'].mean()
    print(">> accuracy: {:.4f}\n".format(accuracy))

    with open(config['file_result'], 'w') as f:
        f.write("# accuracy: {:.4f}\n".format(accuracy))
        df_result.to_csv(f, index=False)


if __name__ == '__main__':

    main()
