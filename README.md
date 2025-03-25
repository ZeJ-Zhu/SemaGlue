Research @  (AAAI 2025)

# SemaGlue Inference and Evaluation Demo Script

## Introduction

Pytorch implementation of SemaGlue for AAAI 2025 paper "Matching While Perceiving: Enhance Image Feature Matching with Applicable Semantic Amalgamation".

Image feature matching is a cardinal problem in computer vision, aiming to establish accurate correspondences between two-view images. Existing methods are constrained by the performance of feature extractors and struggle to capture local information affected by sparse texture or occlusions. Recognizing that human eyes consider not only similar local geometric features but also high-level semantic information of scene objects when matching images, this paper introduces SemaGlue. This novel algorithm perceives and incorporates semantic information into the matching process. In contrast to recent approaches that leverage semantic consistency to narrow the scope of matching areas, SemaGlue achieves semantic amalgamation with the designed Semantic-Aware Fusion (SAF) Block by injecting abundant semantic features from the pre-trained segmentation model. Moreover, the CrossDomain Alignment (CDA) Block is proposed to address domain alignment issues, bridging the gaps between semantic and geometric domains to ensure applicable semantic amalgamation. 

This repository contains the evaluation for relative pose estimation on YFCC100M dataset and visualization matching results on a pair of images, all of which are described in our paper.

## Dependencies

After creating and activating  a practicable environment, our repository and other dependencies can be easily downloaded and intsalled through git and pip.

* git clone https://github.com/ZeJ-Zhu/SemaGlue.git
* cd SemaGlue
* pip install -r requirements.txt

Then download the pre-trained models from [here](https://drive.google.com/drive/folders/1Mp7BfEWCBDCNXBuUMHGltb8NQgSJQAR0).  All weight files should be saved in .models/weights/.

## Contents

There are two main top-level scripts in this repo:

1.`demo.py` : runs a live demo on image directory

2.`match_pairs.py`: reads image pairs from files and dumps matches to disk (also runs evaluation if ground truth relative poses are provided)

## Matching Demo Script (`demo.py`)

### Visualization mode


<summary>[Click to expand]</summary>

Run the demo on the default given image pairs:

```sh

./demo.py && python demo.py

```

The matches are colored by their predicted confidence in a jet colormap (Green: more confident, Red: less confident).

### Evaluation mode

You can also estimate the pose using RANSAC + Essential Matrix decomposition and evaluate it if the ground truth relative poses and intrinsics are provided in the input `.txt` files. Each `.txt` file contains three key ground truth matrices: a 3x3 intrinsics matrix of image0: `K0`, a 3x3 intrinsics matrix of image1: `K1` , and a 4x4 matrix of the relative pose extrinsics `T_0to1`.

`</details>`

### Recommended settings for YFCC


<summary>[Click to expand]</summary>

For **outdoor** images, we recommend the following settings:

```sh

./match_pairs.py--resize1600--superglueoutdoor--max_keypoints2048--nms_radius3--resize_float

```

You can provide your own list of pairs `--input_pairs` for images contained in `--input_dir`. Images can be resized before network inference with `--resize`. If you are re-running the same evaluation many times, you can use the `--cache` flag to reuse old computation.

### Reproducing the outdoor evaluation on YFCC

<details>

<summary>[Click to expand]</summary>

We provide the groundtruth for YFCC in our format in the file `assets/yfcc_test_pairs_with_gt.txt` for convenience. In order to reproduce similar tables to what was in the paper, you will need to download the dataset (we do not provide the raw test images). To download the YFCC dataset, you can use the [OANet](https://github.com/zjhthu/OANet) repo:

```sh

gitclonehttps://github.com/zjhthu/OANet

cdOANet

bashdownload_data.shraw_dataraw_data_yfcc.tar.gz08

tar-xvfraw_data_yfcc.tar.gz

mvraw_data/yfcc100m~/data

```

Once the YFCC dataset is downloaded in `~/data/yfcc100m`, you can run the following:

```sh

./match_pairs.py--input_dir~/data/yfcc100m--input_pairsassets/yfcc_test_pairs_with_gt.txt--output_dirdump_yfcc_test_results--eval--resize1600--superglueoutdoor--max_keypoints2048--nms_radius3--resize_float

```

You should get the following table for YFCC:

```txt

Evaluation Results (mean over 4000 pairs):

AUC@5    AUC@10  AUC@20  Prec    MScore

40.10    60.35   76.24   99.14   21.72  

```

## BibTeX Citation

If you use any ideas from the paper or code from this repo, please consider citing:

```txt

```
