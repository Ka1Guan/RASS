<div align="center">


<h1>Restoration Adaptation for Semantic Segmentation on Low Quality Images</h1>

<div>
    <a href='https://scholar.google.com/citations?user=oNZzFRIAAAAJ&hl=zh-CN' target='_blank'>Kai Guan<sup>1,2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN' target='_blank'>Rongyuan Wu<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=Bd73ldQAAAAJ&hl=zh-TW' target='_blank'>Shuai Li<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=ZEhMnIMAAAAJ&hl=en' target='_blank'>Wentao Zhu<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=_cUfvYQAAAAJ&hl=zh-CN' target='_blank'>Wenjun Zeng<sup>2,† </sup></a>
    <a href='https://www4.comp.polyu.edu.hk/~cslzhang/' target='_blank'>Lei Zhang<sup>1,2,† </sup></a>
</div>
<div>
    <sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>Eastern Institute of Technology, Ningbo&emsp; 
</div>

[[paper]]()

---

</div>

## 🎬 Overview
![overview](asserts/rass.png)

## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/Ka1Guan/RASS.git
    cd RASS
    ```

2. Install dependent packages
    ```bash
    conda create -n rass python=3.10 -y
    conda activate rass
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install --upgrade pip
    pip install -r requirements.txt

    # build detectron2
    pip install -U opencv-python
    git clone git@github.com:facebookresearch/detectron2.git
    cd detectron2
    pip install -e .
    pip install git+https://github.com/cocodataset/panopticapi.git
    pip install git+https://github.com/mcordts/cityscapesScripts.git
    cd ..
    git clone git@github.com:facebookresearch/Mask2Former.git
    cd Mask2Former
    pip install -r requirements.txt
    cd mask2former/modeling/pixel_decoder/ops
    sh make.sh
    ```

3. Download Models
* [RASS](https://drive.google.com/file/d/14Ut18YU2OJ-J1OQHOQK6NTs-4PQYvrJp/view?usp=sharing)

#### Dependent Models
* [SD21 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
* [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)
* [DAPE](https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link)


## ⚡ Quick Inference
```
python train_net.py --config-file configs/ade20k/semantic-segmentation/maskformer2_RASS.yaml \
--eval-only MODEL.WEIGHTS preset/model/rass.pth \
OUTPUT_DIR ./experiments
```

##  Training for RASS
Before starting training, you need to prepare the dataset. Please refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md) to download and deploy ADE20K. Then download the simulated degradation validation set ([here](#dataset)) to replace the original standard quality validation images. During training, input the standard quality images and simulate degradation using online data preprocessing().
```
python train_net.py --config-file configs/ade20k/semantic-segmentation/maskformer2_RASS.yaml --num-gpus 8
```

##  Dataset
Simulated degradation of ADE20K test images:
* [ADE20K](https://drive.google.com/file/d/1w_3fMVfwMEJs1Y9pPPknW2gqa33JbtNf/view?usp=sharing)

Real-world low-quality images and annotations:
* [RealLQ](https://drive.google.com/file/d/1ZjA0Vr5kPgSaHrnUHQNcTX5FRQNuybFD/view?usp=sharing)

