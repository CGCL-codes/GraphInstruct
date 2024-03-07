# GTG: Graph Tasks Generation for LLMs

This is the benchmark proposed in our paper: **Empowering Large Language Models with Graph Understanding and Reasoning Capability**

# Install

GTG can be installed with pip:

```
cd GTG
pip install -e .
```

# Dataset Generation

We provide an example script to generate data for all the tasks: `GTG/script/run_all_generation.sh`. 
You only need to modify the `project_root` in the script to your own path, and run:

```
bash run_all_generation.sh
```

Then you'll find the generated dataset in `GTG/data/dataset`. 


# Evaluation

We provide scripts for evaluation (see `GTG/script/evaluation` and `GTG/script/run_all_evaluation.py`). 
The input data file (i.e. LLM's output) should be a csv with 2 columns: `id` (sample ID) and `output` (LLM's output text). 
For example: 

```
id,output
12,"node 5"
9,"node 33"
33,"node 10"
```
