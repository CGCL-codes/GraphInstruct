# NOTE: This script uses relative paths

task_names=(
    'shortest_path'
    'page_rank'
    'DFS'
    'BFS'
    'degree'
    'common_neighbor'
    'jaccard'
    'connectivity'
    'topological_sort'
    'cycle'
    'edge'
    'neighbor'
    'predecessor'
    'bipartite'
    'clustering_coefficient'
    'diameter'
    'MST'
    'euler_path'
    'hamiltonian_path'
    'maximum_flow'
    'connected_component'
)

for task_name in "${task_names[@]}"
do
    new_file_name="../evaluation/${task_name}-eval.sh"
    sed "s/TASK_NAME/${task_name}/g" TASK_NAME-eval.sh > ${new_file_name}
    # echo "bash \$script_root/${task_name}-eval.sh \$data_root/${task_name} \$output_root" >> ../run_all_evaluation.sh
done
