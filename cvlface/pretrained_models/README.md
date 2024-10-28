
# Face Recognition 

## Face Recognition Performance Board

| Arch          | Loss         | Dataset    | Link                                                                          | AVG   | LFW   | CPFLW | CFPFP | CALFW | AGEDB | IJBB@0.01 | IJBC@0.01 | TinyFace R1 | TinyFace R5 |
|---------------|--------------|------------|-------------------------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-----------|-----------|-------------|-------------|
| ViT KPRPE [1] | AdaFace [2]  | WebFace12M | To be released on Aug.                                                        | 93.13 | 99.82 | 95.65 | 99.30 | 95.93 | 98.10 | 96.55     | 97.82     | 76.10       | 78.92       |
| ViT KPRPE [1] | AdaFace [2]  | WebFace4M  | [🤗](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface4m) | 92.76 | 99.83 | 95.40 | 99.01 | 96.00 | 97.67 | 95.56     | 97.13     | 75.75       | 78.49       |
| ViT       [1] | AdaFace [2]  | WebFace4M  | [🤗](https://huggingface.co/minchul/cvlface_adaface_vit_base_webface4m)       | 92.48 | 99.80 | 94.97 | 98.94 | 96.03 | 97.48 | 95.60     | 97.14     | 74.79       | 77.58       |
| IR101     [3] | AdaFace [2]  | WebFace12M | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m)         | 92.13 | 99.82 | 94.57 | 99.24 | 96.12 | 98.00 | 96.46     | 97.72     | 72.42       | 74.81       |
| IR101     [3] | AdaFace [2]  | WebFace4M  | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m)          | 91.98 | 99.83 | 94.63 | 99.27 | 96.05 | 97.90 | 96.10     | 97.46     | 72.13       | 74.49       |
| IR101     [3] | Arc-Face [3] | WebFace4M  | [🤗](https://huggingface.co/minchul/cvlface_arcface_ir101_webface4m)          | 91.76 | 99.78 | 94.35 | 99.21 | 96.00 | 97.95 | 95.83     | 97.30     | 71.03       | 74.41       |
| IR101     [3] | AdaFace [2]  | MS1MV3     | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir101_ms1mv3)             | 90.99 | 99.83 | 93.92 | 99.09 | 96.02 | 98.18 | 95.82     | 97.05     | 67.95       | 71.03       |
| IR101     [3] | AdaFace [2]  | MS1MV2     | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir101_ms1mv2)             | 90.90 | 99.80 | 93.53 | 98.61 | 96.12 | 98.05 | 95.59     | 96.81     | 68.11       | 71.49       |
| IR50      [3] | AdaFace [2]  | MS1MV2     | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir50_ms1mv2)              | 89.96 | 99.85 | 92.85 | 98.09 | 96.07 | 97.85 | 94.86     | 96.20     | 64.99       | 68.88       |
| IR50      [3] | AdaFace [2]  | WebFace4M  | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m)           | 91.48 | 99.78 | 94.17 | 98.99 | 95.98 | 97.78 | 95.49     | 97.01     | 70.20       | 73.93       |
| IR50      [3] | AdaFace [2]  | CASIA      | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir50_casia)               | 77.43 | 99.37 | 90.02 | 97.04 | 93.43 | 94.40 | 46.04     | 52.97     | 59.44       | 64.14       |
| IR18      [3] | AdaFace [2]  | WebFace4M  | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir18_webface4m)           | 89.55 | 99.58 | 92.28 | 97.80 | 95.52 | 96.48 | 92.75     | 94.79     | 66.07       | 70.71       |
| IR18      [3] | AdaFace [2]  | VGG2       | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir18_vgg2)                | 88.12 | 99.53 | 91.73 | 97.64 | 93.90 | 94.07 | 90.07     | 92.40     | 64.62       | 69.15       |
| IR18      [3] | AdaFace [2]  | CASIA      | [🤗](https://huggingface.co/minchul/cvlface_adaface_ir18_casia)               | 72.40 | 99.22 | 87.00 | 94.93 | 92.65 | 92.68 | 30.36     | 37.10     | 56.20       | 61.43       |

   * [1] [KP-RPE: KeyPoint Relative Position Encoding for Face Recognition](https://arxiv.org/abs/2403.14852)
   * [2] [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)
   * [3] [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
   * [4] [LFW: Labeled faces in the wild: A database forstudying face recognition in unconstrained environments](https://vis-www.cs.umass.edu/lfw/lfw.pdf)
   * [5] [CPFLW: Cross-pose lfw: A database for studying cross-pose face recognition in unconstrained environments](http://www.whdeng.cn/cplfw/?reload=true)
   * [6] [CFPFP: Frontal to profile face verification in the wild](https://ieeexplore.ieee.org/document/7477558)
   * [7] [CALFW: Cross-Age LFW: A Database for Studying Cross-Age Face Recognition in Unconstrained Environments](https://arxiv.org/abs/1708.08197)
   * [8] [AGEDB: The First Manually Collected, In-the-Wild Age Database](https://ieeexplore.ieee.org/document/8014984)
   * [9] [IJBB: IARPA Janus Benchmark-B Face Dataset](https://ieeexplore.ieee.org/document/8014821)
   * [10] [IJBC: IARPA Janus Benchmark-C Face Dataset and Protocol](https://ieeexplore.ieee.org/document/8411217)
   * [11] [TinyFace: Low-resolution face recognition](https://arxiv.org/abs/1811.08965) 



## Replicate Evaluation Scores with Pretrained Models

The table can be replicated by running the following command:
1. Download pretrained models by following the instruction [README_MODELS.md](../../README_MODELS.md) 
2. Run command
```bash
cd cvlface/research/recognition/code/run_v1/
bash scripts/eval/run_multi_gpu.sh
```

