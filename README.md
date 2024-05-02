# Language Driven Object Grasping via Feature Augmentation

# Abstract 
To address the challenge of few-shot learning in language-driven grasp detection, we introduce GraspLLaMA. This approach enhances data robustness through augmentation and leverages transfer learning to effectively train on limited datasets.

## Setup
First, clone the LLaVa repository and install the dependencies:
```bash
git clone https://github.com/haotian-liu/LLaVA
cd LLaVA
pip3 install -e .
```

Then run `pip install -r requirements.txt` to install the required dependencies.

Download dataset[GraspAnything++](https://drive.google.com/file/d/1h2_x18WtR9N58e0O14BNbo_5AI6yWgeu/view) and store it in grasp-anything++ folder.

## Training

Then, run LoRA fine-tuning with generated dataset for LLaVa to adapt with [SPT] tokens generation. If you want to generate the instruction dataset from scratch, you might want to care about `scripts/synthetic_llava_format.py` and `scripts/synthetic.py` to generate the dataset.
```bash
bash scripts/train_llava.sh
```

After that, you can train the LLaVa-Grasp model with the following command:
```bash
python3 main.py --use-depth=0
```

## Evaluation
To evaluate the model, you can run the following command:
```bash
python3 main.py --use-depth=0 --test --pretrained_path /path/to/pretrained/model
```
