# modify this path:
project_root=/home/songxiran/GraphLLM_code_and_data

script_root=$project_root/xiran-GraphLLM/GTG/script/evaluation
data_root=/home/songxiran/GraphLLM_code_and_data/data/dataset/0118/small

# modify this path:
output_root=$1

# bash $script_root/shortest_path-eval.sh $data_root/shortest_path $output_root
# bash $script_root/page_rank-eval.sh $data_root/page_rank $output_root
# bash $script_root/DFS-eval.sh $data_root/DFS $output_root
# bash $script_root/degree-eval.sh $data_root/degree $output_root
bash $script_root/common_neighbor-eval.sh $data_root/common_neighbor $output_root
# bash $script_root/jaccard-eval.sh $data_root/jaccard $output_root
# bash $script_root/connectivity-eval.sh $data_root/connectivity $output_root
# bash $script_root/topological_sort-eval.sh $data_root/topological_sort $output_root
# bash $script_root/edge-eval.sh $data_root/edge $output_root
# bash $script_root/neighbor-eval.sh $data_root/neighbor $output_root
# bash $script_root/predecessor-eval.sh $data_root/predecessor $output_root
# bash $script_root/bipartite-eval.sh $data_root/bipartite $output_root
# bash $script_root/diameter-eval.sh $data_root/diameter $output_root
# bash $script_root/MST-eval.sh $data_root/MST $output_root
# bash $script_root/hamiltonian_path-eval.sh $data_root/hamiltonian_path $output_root
# bash $script_root/maximum_flow-eval.sh $data_root/maximum_flow $output_root
# bash $script_root/connected_component-eval.sh $data_root/connected_component $output_root

# bash $script_root/BFS-eval.sh $data_root/BFS $output_root
# bash $script_root/clustering_coefficient-eval.sh $data_root/clustering_coefficient $output_root
# bash $script_root/cycle-eval.sh $data_root/cycle $output_root
# bash $script_root/euler_path-eval.sh $data_root/euler_path $output_root
