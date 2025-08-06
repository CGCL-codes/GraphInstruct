model_name_or_path=/home/ud202281368/data/base_model/llama3.1-8b-instruct
adapter_name_or_path=/home/ud202281368/jupyterlab/LLaMA-Factory/saves/llama3.1-8b/lora/sft
datasets=(
  "BFS-test"
  "bipartite-test"
  "clustering_coefficient-test"
  "common_neighbor-test"
  "connected_component-test"
  "connectivity-test"
  "cycle-test"
  "degree-test"
  "DFS-test"
  "diameter-test"
  "edge-test"
  "euler_path-test"
  "hamiltonian_path-test"
  "jaccard-test"
  "maximum_flow-test"
  "MST-test"
  "neighbor-test"
  "page_rank-test"
  "predecessor-test"
  "shortest_path-test"
  "topological_sort-test"
)

suffix=prediction.jsonl

for dataset in "${datasets[@]}"; do
  python evaluation/graphinstruct.py \
      --model_name_or_path $model_name_or_path \
      --adapter_name_or_path $adapter_name_or_path \
      --dataset $dataset \
      --template llama3 \
      --save_name ${dataset}${suffix}
done
