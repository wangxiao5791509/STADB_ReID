# STADB_ReID
Official implementation of "STADB: A Self-Thresholding Attention Guided ADB Network for Person Re-identification", Bo Jiang, Sheng Wang, Xiao Wang, Aihua Zheng [[arxiv](https://arxiv.org/abs/2007.03584)] 

## Background and Motivation
Recently, Batch DropBlock network (BDB) has demonstrated its effectiveness on person image representation and re-identification task via feature erasing. However, BDB drops the features randomly which may lead to sub-optimal results. In this paper, we propose a novel Self-Thresholding attention guided Adaptive DropBlock network (STADB) for person re-ID which can adaptively erase the most discriminative regions. Specifically, STADB first obtains an attention map by channel-wise pooling and returns a drop mask by thresholding the attention map. Then, the input features and self-thresholding attention guided drop mask are multiplied to generate the dropped feature maps. In addition, STADB utilizes the spatial and channel attention to learn a better feature map and iteratively trains the feature dropping module for person re-ID. Experiments on several benchmark datasets demonstrate that the proposed STADB outperforms many other related methods for person re-ID. 


## Framework 


## Environment 



## Training 




## Testing 



## Citation 
If you find this work useful for your research, please cite this paper: 
```
@article{bojiang2020stadb,
  title={STADB: A Self-Thresholding Attention Guided ADB Network for Person Re-identification},
  author={Bo Jiang, Sheng Wang, Xiao Wang, Aihua Zheng},
  journal={arXiv:2007.03584},
  year={2020}
}
```





























