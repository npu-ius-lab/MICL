# MICL

Code for the paper "MICL: Mutual Information Guided Continual Learning for LiDAR Place Recognition".

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
