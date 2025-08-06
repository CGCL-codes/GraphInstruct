
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_graphinstruct/llama3_lora_sft.yaml

llamafactory-cli export examples/merge_graphinstruct/llama3_lora_sft.yaml

bash evaluation/evaluation.sh