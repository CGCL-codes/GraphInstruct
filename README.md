# GraphInstruct

This is the benchmark proposed in our paper: **GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability**

## Dataset Generation and Evaluation

As a dynamic dataset, GraphInstruct can be generated from scratch and used for evaluation with the following steps:

### Environment Install

GTG can be installed with pip:

```
cd GTG
pip install -e.
```
> [!IMPORTANT]
> Installation is mandatory.

### Dataset Generation

We provide an example script to generate data for all the tasks: `GTG/script/run_all_generation.sh`. 
You only need to modify the `project_root` in the script to your own path, and run:

```
bash run_all_generation.sh
```

Then you'll find the generated dataset in `GTG/data/dataset`. 


### Evaluation

We provide scripts for evaluation (see `GTG/script/evaluation` and `GTG/script/run_all_evaluation.py`). 
The input data file (i.e. LLM's output) should be a csv with 2 columns: `id` (sample ID) and `output` (LLM's output text). 
For example: 

```
id,output
12,"node 5"
9,"node 33"
33,"node 10"
```

## Model Training

Our implementation for training GraphSolver and GraphSolver+ is mainly based on [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory).

### Dataset Preparation

- Due to space limitation, we only provide our training json files for GraphSolver+ in `LLaMAFactory/data/reasoning`. 

- For getting detailed dataset files, one can refer to the **Dataset Generation** step in GTG.

### Supervised Fine-tuning

One can start the model training step with the following command:

```
cd LLaMAFactory
bash run.sh
```

Note that, to ensure proper functioning, it is necessary to adjust the experiment settings in `examples/train_reasoning/llama3_lora_sft.yaml` and `examples/merge_reasoning/llama3_lora_sft.yaml`.

> [!TIP]
> For more details about the experimental configuration, please refer to the [readme.md](https://github.com/CGCL-codes/GraphInstruct/blob/main/LLaMAFactory/data/README.md) in LLaMAFactory.

