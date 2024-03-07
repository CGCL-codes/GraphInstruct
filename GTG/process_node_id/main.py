import GTG
from GTG.utils.parse_arguments import parse_arguments
from GTG.utils.utils import set_random_seed

from tqdm import tqdm
import pandas as pd


def main():
    
    config = parse_arguments()
    
    if 'seed' in config:
        set_random_seed(config['seed'])

    Mapping = {
        'int_id': GTG.process_node_id.int_id.Mapping,
        'letter_id': GTG.process_node_id.letter_id.Mapping,
    }[config['id_type']]

    df = pd.read_csv(config['file_input'])
    df_dict = {key: [] for key in df.keys()}
    for i in tqdm(range(len(df))):
        df_dict['task'].append(df['task'][i])
        df_dict['num_nodes'].append(df['num_nodes'][i])
        df_dict['num_edges'].append(df['num_edges'][i])
        df_dict['directed'].append(df['directed'][i])
        s_graph = df['graph'][i]
        id_map = Mapping(s_graph)  # init mapping
        df_dict['graph'].append(id_map(df['graph'][i]))
        df_dict['graph_adj'].append(id_map(df['graph_adj'][i]))
        df_dict['graph_nl'].append(id_map(df['graph_nl'][i]))
        df_dict['nodes'].append(id_map(df['nodes'][i]))
        df_dict['question'].append(id_map(df['question'][i]))
        df_dict['answer'].append(id_map(str(df['answer'][i])))
        if 'steps' in df_dict:
            df_dict['steps'].append(id_map(df['steps'][i]))
        if 'choices' in df_dict:
            df_dict['choices'].append(id_map(df['choices'][i]))
        if df['task'][i]=='MST':
            df_dict['answer_tree'].append(id_map(df['answer_tree'][i]))
        if 'label' in df_dict:
            df_dict['label'].append(df['label'][i])
        if 'n1' in df_dict:
            df_dict['n1'].append(df['n1'][i])
            df_dict['n2'].append(df['n2'][i])
        df_dict['id'].append(df['id'][i])
        if 'cc_node_ratio' in df_dict:
            df_dict['cc_node_ratio'].append(df['cc_node_ratio'][i])

    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(config['file_output'], index=False)


if __name__ == '__main__':

    main()
