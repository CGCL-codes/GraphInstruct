# modify this path:
project_root=/Users/songxiran/code/GraphLLM_code_and_data/GraphLLM/GTG

script_root=$project_root/script/dataset_generation

data_root=$project_root/data/dataset/toy/$dataset_tag

################################
dataset_tag=mini

mkdir -p $data_root

num_sample=300
num_nodes_range="(5,7)"

bash $script_root/shortest_path.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/page_rank.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/DFS.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/BFS.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/degree.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/common_neighbor.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/jaccard.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/connectivity.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/topological_sort.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/cycle.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/edge.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/neighbor.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/predecessor.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/bipartite.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/clustering_coefficient.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/diameter.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/MST.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/euler_path.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/hamiltonian_path.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/maximum_flow.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/connected_component.sh $data_root $num_nodes_range $num_sample $dataset_tag


################################
dataset_tag=small
data_root=$project_root/data/dataset/0118/$dataset_tag

# rm -r $data_root
mkdir -p $data_root

source /home/xiran/env/LLM/bin/activate

num_sample=300
num_nodes_range="(8,15)"

bash $script_root/shortest_path.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/page_rank.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/DFS.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/BFS.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/degree.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/common_neighbor.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/jaccard.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/connectivity.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/topological_sort.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/cycle.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/edge.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/neighbor.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/predecessor.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/bipartite.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/clustering_coefficient.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/diameter.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/MST.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/euler_path.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/hamiltonian_path.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/maximum_flow.sh $data_root $num_nodes_range $num_sample $dataset_tag
bash $script_root/connected_component.sh $data_root $num_nodes_range $num_sample $dataset_tag
