# MICL

Code for the paper "MICL: Mutual Information Guided Continual Learning for LiDAR Place Recognition".
# Abstract
LiDAR Place Recognition (LPR) aims to identify previously visited places across different environments and times. Thanks to the recent advances in Deep Neural Networks (DNNs), LPR has experienced rapid development. However, DNN-based LPR methods may suffer from Catastrophic Forgetting (CF), where they tend to forget previously learned domains and focus more on adapting to a new domain. In this paper, we propose Mutual Information-guided Continual Learning (MICL) to tackle this problem in LPR. We design a domain-sharing loss function Mutual Information Loss (MIL) to encourage existing DNN-based LPR methods to learn and preserve knowledge that may not be useful for the current domain but potentially beneficial for other domains. MIL overcomes CF from an information-theoretic perspective including two aspects:1) maximizing the preservation of information from input data in descriptors, and 2) maximizing the preservation of information in descriptors when training across different domains. Additionally, we design a simple yet effective memory sampling strategy to further alleviate CF in LPR. Furthermore, we adopt adaptive loss weighting, which reduces the need for hyperparameters and enables models to make optimal trade-offs automatically. We conducted experiments on three large-scale LiDAR datasets including Oxford, MulRan, and PNV. The experimental results demonstrate that our MICL outperforms state-of-the-art continual learning approaches. 

## Installation
Code was tested using Python 3.8 with PyTorch 1.12.1 and MinkowskiEngine 0.5.4 on Ubuntu 20.04 with CUDA 11.3

The following Python packages are required:
* PyTorch (version 1.12.1)
* MinkowskiEngine (version 0.5.4)
* pytorch_metric_learning (version 1.0 or above)
* torchpack
* tensorboard
* pandas


Modify the `PYTHONPATH` environment variable to include an absolute path to the project root folder: 
```
export PYTHONPATH=$PYTHONPATH:/.../.../Metric_learning_based
```

## Data Preparation 
We follow this [link](https://github.com/csiro-robotics/InCloud) to generate .pickle files.

## Training
We provide a launch.json file in the training folder, which contains two pre-configured training commands. You can adapt the content of this file to fit your own project needs.

### Evaluation
To evaluate InCloud run the following command:

    python eval/evaluate.py --config config/protocols/<config> --ckpt <path_to_ckpt>
   
   Where `<config>` and `<path_to_ckpt>` are the config file for the evaluation method you wish to evaluate and the path to the checkpoint you wish to evaluate. 

## Pretrained Models

The following models from the paper are provided for evaluation purposes, all the models are firstly trained on the oxford dataset and continually trained on the existing three ones. The results of different backbone networks using Mean recall@1 metric are as follows:

### Metric learning-based MICL

| Architecture | Mean Recall@1 
|--------------|---------------
| MinkLoc3D    | 85.9          
| LoGG3D-Net   | 80.3          
| PointNetVLAD | 65.6          

### Contrastive learning-based MICL

| Architecture | Mean Recall@1 
|--------------|---------------
| MinkLoc3D    | 84.1          
| LoGG3D-Net   | 85.2          
| PointNetVLAD | 68.2          

Checkpoints Link: [https://drive.google.com/drive/folders/1l2v7Jw3PpIQmN5jjrZntPwzHwTVT5JD7?usp=drive_link](https://drive.google.com/drive/folders/1l2v7Jw3PpIQmN5jjrZntPwzHwTVT5JD7?usp=drive_link)

## Cite
If you find this repository useful for your research, please consider citing the paper

```
@ARTICLE{10713116,
  author={Liu, Binhong and Yang, Tao and Fang, Yangwang and Yan, Zhi},
  journal={IEEE Robotics and Automation Letters}, 
  title={MICL: Mutual Information Guided Continual Learning for LiDAR Place Recognition}, 
  year={2024},
  volume={9},
  number={11},
  pages={10463-10470},
  doi={10.1109/LRA.2024.3475031}
}
```
## Acknowledgements
We would like to acknowledge the authors of [InCloud](https://github.com/csiro-robotics/InCloud) and [CCL](https://github.com/cloudcjf/CCL) for their excellent codebases.


