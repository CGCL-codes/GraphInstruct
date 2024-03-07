data_root=$1
num_nodes_range=$2
num_sample=$3
dataset_tag=$4

task_name=maximum_flow

root=$data_root/$task_name
mkdir -p $root

file_output_raw=$root/$task_name.csv

python -m GTG.generation --task $task_name \
    --file_output $file_output_raw \
    --num_samples $num_sample \
    --num_nodes_range $num_nodes_range \
    --hash_str $dataset_tag \


file_input=$file_output_raw
id_type='int_id'
file_output=$root/$task_name-${id_type}.csv

python -m GTG.process_node_id.main --id_type $id_type \
    --file_input $file_input \
    --file_output $file_output \


file_input=$file_output_raw
id_type='letter_id'
file_output=$root/$task_name-${id_type}.csv

python -m GTG.process_node_id.main --id_type $id_type \
    --file_input $file_input \
    --file_output $file_output \
