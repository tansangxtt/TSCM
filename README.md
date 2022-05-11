# Robust Neural Network for Real-Time Object Segmentation
This reporsitory is for our term project of Machine Learning COMP5212.

Our group has four members: 
TRUONG Quang Trung,
HA Tan Sang,
NGUYEN Huu Canh,
Mingzhi SHIHUA.

## Installation
Create a conda environment and install the following dependencies:
```shell script
sudo apt install ninja-build  # For Debian/Ubuntu
conda install -y cython pip scipy scikit-image tqdm
conda install -y pytorch torchvision cudatoolkit -c pytorch
pip install opencv-python easydict
```

## Datasets
### DAVIS

To test the DAVIS validation split, download and unzip the 2017 480p trainval images and annotations here:
[file](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip).

### YouTubeVOS

To test our validation split and the YouTubeVOS challenge 'valid' split, download [YouTubeVOS 2018](https://youtube-vos.org/dataset/)
and place it in this directory structure:

```
/path/to/ytvos2018
|-- train/
|-- train_all_frames/
|-- valid/
`-- valid_all_frames/
```

## Models

These pretrained models are available for download: 

| Backbone  | Training set       | Weights  |
|:---------:|:------------------:|:--------:|
| ResNet101  | DAVIS         | [download](https://hkustconnect-my.sharepoint.com/:u:/g/personal/qttruong_connect_ust_hk/EQ9vU8M0yR9GsNXJ3NQzz00BfJ32QUeBWl2ys01uaqSBfA?e=0XFB8Q) |
| ResNet18  | DAVIS | [download](https://hkustconnect-my.sharepoint.com/:u:/g/personal/qttruong_connect_ust_hk/EbTm1WCgyVhOrenseX2RxrIBYVF7PuGyJ8HxXWcmqE6Vaw?e=ShG7Kg)|
| Other methods | DAVIS and YouTubeVOS              | [download](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qttruong_connect_ust_hk/ErGD3CriQ9lNlkwJNHcFfLkBGBd4SW5p2dMmWuhSaNz8iw?e=w0ixKs) |


Download pth files fom the [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qttruong_connect_ust_hk/ErGD3CriQ9lNlkwJNHcFfLkBGBd4SW5p2dMmWuhSaNz8iw?e=JdEWSs) and put them in the folder `weights/`.

## Training

### How to run the training code

Training is set up similarly to evaluation.

Open `train.py` and adjust the `paths` dict to your dataset locations, checkpoint and tensorboard
output directories and the place to cache target model weights.

Shell script for training a network:

```shell script
python train.py --ftext resnet101 --dset all --dev cuda:0
```
`--ftext` is the name of the feature extractor, either resnet18 or resnet101.

`--dset` is one of dv2017, ytvos2018 or all ("all" really means "both").

`--dev` is the name of the device to train on.

## How to run the evaluation code

Open `evaluate.py` and adjust the `paths` dict to your dataset locations and where you want the output.
The dictionary is found near line 110, and looks approximately like this:

```python
    paths = dict(
        models=Path(__file__).parent / "weights",  # The .pth files should be here
        davis="/path/to/DAVIS",  # DAVIS dataset root
        yt2018="/path/to/ytvos2018",  # YouTubeVOS 2018 root
        output="/path/to/results",  # Output path
    )
```

Shell script for evaluation:
```shell script
python evaluate.py --model file.pth --dset dataset_directory
```

`--model` is the name of the checkpoint to use in the `weights` directory.

`--fast` reduces the number of optimizer iterations to match "Ours fast" in the paper.

`--dset` is one of

  | Name        | Description                                                |
  |-------------|------------------------------------------------------------|
  | dv2016val   | DAVIS 2016 validation set                                  |
  | dv2017val   | DAVIS 2017 validation set                                  |
  | yt2018jjval | Our validation split of YouTubeVOS 2018 "train_all_frames" |
  | yt2018val   | YouTubeVOS 2018 official "valid_all_frames" set            |
