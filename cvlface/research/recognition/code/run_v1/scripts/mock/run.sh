
# 1 gpu
python train.py trainers.prefix=test_run \
      trainers.num_gpu=1 \
      trainers.batch_size=32 \
      trainers.limit_num_batch=128 \
      trainers.gradient_acc=1 \
      trainers.num_workers=8 \
      trainers.precision='32-true' \
      trainers.float32_matmul_precision='high' \
      dataset=configs/synthetic.yaml \
      data_augs=configs/basic_v1.yaml \
      models=iresnet/configs/v1_ir50.yaml \
      pipelines=configs/train_model_cls.yaml \
      evaluations=configs/base.yaml \
      classifiers=configs/fc.yaml \
      optims=configs/step_sgd.yaml \
      losses=configs/cosface.yaml

# multi gpu
LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 lightning run model \
    --strategy=ddp \
    --devices=7 \
    --precision="32-true" \
      train.py trainers.prefix=test_run \
      trainers.num_gpu=7 \
      trainers.batch_size=32 \
      trainers.limit_num_batch=128 \
      trainers.gradient_acc=1 \
      trainers.num_workers=8 \
      trainers.precision='32-true' \
      trainers.float32_matmul_precision='high' \
      dataset=configs/synthetic.yaml \
      data_augs=configs/basic_v1.yaml \
      models=iresnet/configs/v1_ir50.yaml \
      pipelines=configs/train_model_cls.yaml \
      evaluations=configs/base.yaml \
      classifiers=configs/fc.yaml \
      optims=configs/step_sgd.yaml \
      losses=configs/cosface.yaml