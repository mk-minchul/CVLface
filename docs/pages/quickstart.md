# Quick start

<p align="center">
 🌎 <a href="https://github.com/mk-minchul/CVLface" target="_blank">GitHub</a> • 🤗 <a href="https://huggingface.co/minchul" target="_blank">Hugging Face</a> 
</p>

## Quick Installation

#### 1. Download the repository
You can install CVLface via git clone.

```bash
git clone https://github.com/cvl/cvlface.git CVLface
cd CVLface
pip install -e .
```
This allows you to install the package in editable mode, so you can modify the source code and see the changes immediately.

#### 2. Setup the Environment Variables

You should also modify the file `cvlface/.env`. Initially it looks like:

```bash
DATA_ROOT="path where datasets will be stored"
HF_TOKEN="token for https://huggingface.co/"
WANDB_TOKEN="token for https://wandb.ai/"
```

Modify the variables to your own.

You are done!

## Train from Scratch

To train a face recognition model, you can use the following command: We use `lightning` to go multi-GPU.

```bash
# go to face recognition code path
cd cvlface/research/recognition/code/run_v1

# example command
LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 lightning run model \
    --strategy=ddp \
    --devices=7 \
    --precision="32-true" \
      train.py trainers.prefix=ir101_WF4M_adaface \
      trainers.num_gpu=7 \
      trainers.batch_size=256 \
      trainers.gradient_acc=1 \
      trainers.num_workers=8 \
      trainers.precision='32-true' \
      trainers.float32_matmul_precision='high' \
      dataset=configs/webface4m.yaml \
      data_augs=configs/basic_v1.yaml \
      models=iresnet/configs/v1_ir101.yaml \
      pipelines=configs/train_model_cls.yaml \
      evaluations=configs/full.yaml \
      classifiers=configs/fc.yaml \
      optims=configs/step_sgd.yaml \
      losses=configs/adaface.yaml \
      trainers.skip_final_eval=False
```

The command takes care of downloading necessary datasets to the `$DATA_ROOT` directory. 

The model logs and output will be saved in the `cvlface/research/recognition/experiments/run_v1` directory.

