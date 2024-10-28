# 🌟 Setting Up the Training Dataset for CVLFace

<p align="center">
 🌎 <a href="https://github.com/mk-minchul/CVLface" target="_blank">GitHub</a> • 🤗 <a href="https://huggingface.co/minchul" target="_blank">Hugging Face</a> 
</p>

This guide provides detailed instructions on how to download and configure the training datasets necessary for training face recognition models using the CVLFace toolkit. We utilize `.rec` files, which are similar to HDF5 files and are optimized for high-performance read and write operations.

## Prerequisites 🛠️

Before beginning, make sure you have configured the `$DATA_ROOT` directory in your `cvlface/.env` file. 
This directory will serve as the base path for all dataset directories.

## Dataset Structure 🗂️

The datasets are neatly organized into directories where the name pertains to the name in the data config's yaml file.

For example, the `cvlface/research/recognition/code/run_v1/dataset/configs/casia.yaml` file contains the following configuration:
```yaml
yaml file: 
data_root: ${oc.env:DATA_ROOT}
rec: 'casia_webface'
color_space: 'RGB'
num_classes: 10572
num_image: 490623
repeated_sampling_cfg: null
semi_sampling_cfg: null
```
Then the `rec` **field** in the yaml file corresponds to the directory name in the `$DATA_ROOT` directory.

Below is the example directory structure after setup:

### Directory Structure

```plaintext
$DATA_ROOT
├── casia_webface
│   ├── train.rec
│   ├── train.idx
│   └── train.tsv
├── webface260m/WebFace4M
│   ├── train.rec
│   ├── train.idx
│   ├── train.tsv
│   └── ldmk_5points.csv
├── webface260m/WebFace12M
│   ├── train.rec
│   ├── train.idx
│   ├── train.tsv
│   └── ldmk_5points.csv
├── MS1MV2
...
└── custom, etc
...
(Note: You do not need to download all datasets. Select the one(s) that best suit your needs.)
```

## Downloading and Setting Up Datasets

### CASIA-WebFace

1. In terminal, do `DATA_ROOT=/path/to/data_root`
2. `mkdir -p $DATA_ROOT/casia_webface`
1. Download a zipfile from https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view
2. `unzip faces_webface_112x112.zip`
3. `cp faces_webface_112x112/train.rec $DATA_ROOT/casia_webface/train.rec`
4. `cp faces_webface_112x112/train.idx $DATA_ROOT/casia_webface/train.idx`
5. `rm -rf faces_webface_112x112 faces_webface_112x112.zip`
5. (Optional) For training with preprocessed landmarks, (models like KP-RPE), `ldmk_5points.csv` need to be prepared. Run `python predict_landmark.py --source_dir $DATA_ROOT/casia_webface`


### VGG2

1. In terminal, do `DATA_ROOT=/path/to/data_root`
2. `mkdir -p $DATA_ROOT/vgg2`
1. Download a zipfile from https://drive.google.com/file/d/1dyVQ7X3d28eAcjV3s3o0MT-HyODp_v3R/view
2. `unzip faces_vgg_112x112.zip`
3. `cp faces_vgg_112x112/train.rec $DATA_ROOT/vgg2/train.rec`
4. `cp faces_vgg_112x112/train.idx $DATA_ROOT/vgg2/train.idx`
5. `rm -rf faces_vgg_112x112 faces_vgg_112x112.zip`
5. (Optional) For training with preprocessed landmarks, (models like KP-RPE), `ldmk_5points.csv` need to be prepared. Run `python predict_landmark.py --source_dir $DATA_ROOT/vgg2`


### MS1MV2

1. In terminal, do `DATA_ROOT=/path/to/data_root`
2. `mkdir -p $DATA_ROOT/MS1MV2`
1. Download a zipfile from https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view
2. `unzip faces_emore.zip`
3. `cp faces_emore/train.rec $DATA_ROOT/MS1MV2/train.rec`
4. `cp faces_emore/train.idx $DATA_ROOT/MS1MV2/train.idx`
5. `rm -rf faces_emore faces_emore.zip`
5. (Optional) For training with preprocessed landmarks, (models like KP-RPE), `ldmk_5points.csv` need to be prepared. Run `python predict_landmark.py --source_dir $DATA_ROOT/MS1MV2`


### MS1MV3

1. In terminal, do `DATA_ROOT=/path/to/data_root`
2. `mkdir -p $DATA_ROOT/MS1MV3`
1. Download a zipfile from https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view
2. `unzip ms1m-retinaface-t1.zip`
3. `cp ms1m-retinaface-t1/train.rec $DATA_ROOT/MS1MV3/train.rec`
4. `cp ms1m-retinaface-t1/train.idx $DATA_ROOT/MS1MV3/train.idx`
5. `rm -rf ms1m-retinaface-t1 ms1m-retinaface-t1.zip`
5. (Optional) For training with preprocessed landmarks, (models like KP-RPE), `ldmk_5points.csv` need to be prepared. Run `python predict_landmark.py --source_dir $DATA_ROOT/MS1MV3`


### WebFace4M

1. In terminal, do `DATA_ROOT=/path/to/data_root`
2. `mkdir -p $DATA_ROOT/webface260m/temp`
1. Obtain a dataset download link and password from https://www.face-benchmark.org/download.html
2. Download all zipfiles in `0` folder and place them in a folder (ex: 0_0.zip, 0_1.zip, 0_2.zip, etc). `0` folder pertains to WebFace4M split.
3. Unzip all zipfiles into `$DATA_ROOT/webface260m/temp`. Ex) `unzip -d $DATA_ROOT/webface260m/temp 0_1.zip` so on and so forth.
4. Rename folders `mv $DATA_ROOT/webface260m/temp/WebFace260M $DATA_ROOT/webface260m/WebFace4M`
5. `rm -rf $DATA_ROOT/webface260m/temp` and you can remove the zip files as well.
4. Bundle all images into a rec file by running `python bundle_images_into_rec.py --source_dir $DATA_ROOT/webface260m/WebFace4M --remove_images`
5. (Optional) For training with preprocessed landmarks, (models like KP-RPE), `ldmk_5points.csv` need to be prepared. Run `python predict_landmark.py --source_dir $DATA_ROOT/webface260m/WebFace4M`


### WebFace12M

1. In terminal, do `DATA_ROOT=/path/to/data_root`
2. `mkdir -p $DATA_ROOT/webface260m/temp`
1. Obtain a dataset download link and password from https://www.face-benchmark.org/download.html
2. Download all zipfiles in `0,1,2` folder and place them in a folder (ex: 0_0.zip, 0_1.zip, 0_2.zip, etc). `0,1,2` folder pertains to WebFace4M split.
3. Unzip all zipfiles into `$DATA_ROOT/webface260m/temp`. Ex) `unzip -d $DATA_ROOT/webface260m/temp 0_1.zip` so on and so forth.
4. Rename folders `mv $DATA_ROOT/webface260m/temp/WebFace260M $DATA_ROOT/webface260m/WebFace12M`
5. `rm -rf $DATA_ROOT/webface260m/temp` and you can remove the zip files as well.
4. Bundle all images into a rec file by running `python bundle_images_into_rec.py --source_dir $DATA_ROOT/webface260m/WebFace12M --remove_images`
5. (Optional) For training with preprocessed landmarks, (models like KP-RPE), `ldmk_5points.csv` need to be prepared. Run `python predict_landmark.py --source_dir $DATA_ROOT/webface260m/WebFace12M`

