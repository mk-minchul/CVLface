#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir18_casia
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir18_vgg2
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir18_webface4m
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir50_casia
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir50_webface4m
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir50_ms1mv2
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir101_ms1mv2
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir101_ms1mv3
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/arcface_ir101_webface4m
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir101_webface4m
#
#LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
#    --strategy=ddp \
#    --devices=4 \
#    --main_port=9999 \
#    --precision="32-true" \
#    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir101_webface12m

LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
    --strategy=ddp \
    --devices=4 \
    --main_port=9999 \
    --precision="32-true" \
    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_vit_base_webface4m

LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
    --strategy=ddp \
    --devices=4 \
    --main_port=9999 \
    --precision="32-true" \
    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_vit_base_kprpe_webface4m

LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lightning run model \
    --strategy=ddp \
    --devices=4 \
    --main_port=9999 \
    --precision="32-true" \
    eval.py --num_gpu 4 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_vit_base_kprpe_webface12m
