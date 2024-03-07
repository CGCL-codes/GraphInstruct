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
    new_file_name="../dataset_generation/${task_name}.sh"
    sed "s/TASK_NAME/${task_name}/g" TASK_NAME.sh > ${new_file_name}
    # echo "bash \$script_root/${task_name}.sh \$data_root \$num_nodes_range \$num_sample \$dataset_tag" >> ../run_all_generation.sh
done
