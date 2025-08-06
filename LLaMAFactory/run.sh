
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_reasoning/llama3_lora_sft.yaml

llamafactory-cli export examples/merge_reasoning/llama3_lora_sft.yaml