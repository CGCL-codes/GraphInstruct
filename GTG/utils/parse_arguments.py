import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str)
    parser.add_argument("--id_type", type=str)
    parser.add_argument("--file_input", type=str)
    parser.add_argument("--file_output", type=str)
    parser.add_argument("--file_dataset", type=str)
    parser.add_argument("--file_answer", type=str)
    parser.add_argument("--file_result", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hash_str", type=str)
    parser.add_argument("--num_nodes_range", type=str)
    parser.add_argument("--link_prob", type=float)
    parser.add_argument("--num_samples", type=int)

    (args, unknown) = parser.parse_known_args()
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None:
            parsed_results[arg] = '' if value  in ['none', 'None'] else value
    
    return parsed_results
