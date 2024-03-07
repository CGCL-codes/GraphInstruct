data_root=$1
output_root=$2

task_name=degree

file_dataset=$data_root/$task_name-int_id.csv
file_answer=$output_root/$task_name-output.csv
file_result=$output_root/$task_name-results.csv

python -m GTG.evaluation --task $task_name \
    --file_dataset $file_dataset \
    --file_answer $file_answer \
    --file_result $file_result \
