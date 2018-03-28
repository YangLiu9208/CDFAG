# CDFAG

Codes for paper "Transferable Feature Representation for Visible-to-Infrared Cross-Dataset Human Action Recognition"

##Abstract

Recently, infrared human action recognition has attracted increasing attention for it has many advantages over visible light, that is, being robust to illumination change and shadows. However, the infrared action data is limited until now, which degrades the performance of infrared action recognition. Motivated by the idea of transfer learning, an infrared human action recognition framework using auxiliary data from visible light is proposed to solve the problem of limited infrared action data. In the proposed framework, we first construct a novel Cross-Dataset Feature Alignment and Generalization (CDFAG) framework to map the infrared data and visible light data into a common feature space, where Kernel Manifold Alignment (KEMA) and a dual aligned-to-generalized encoders (AGE) model are employed to represent the feature. Then, a support vector machine (SVM) is trained, using both the infrared data and visible light data, and can classify the features derived from infrared data. The proposed method is evaluated on InfAR, which is a publicly available infrared human action dataset. To build up auxiliary data, we set up a novel visible light action dataset XD145. Experimental results show that the proposed method can achieve state-of-the-art performance compared with several transfer learning and domain adaptation methods.

## Model

![fig](J:\Github repository\CDFAG\Fig1.jpg)Figure 1: Illustration of aligned-to-generalized encoders generalizing aligned features across visible light and infrared action datasets.

## Experimental results

![Fig2](J:\Github repository\CDFAG\Fig2.png)

More experimental results and discussions can be found in paper.

## Dataset

InfAR dataset can be downloaded [here](https://sites.google.com/site/gaochenqiang/publication/infrared-action-dataset).

XD145 dataset can be downloaded [here](https://sites.google.com/site/yangliuxdu/home).

## Codes

CDFAG demo codes can be downloaded here.

Just run "CDFAG_demo.m".

Any problems can be contacted with aryanr@163.com.



If you use our codes or dataset and find our work useful, please kindly consider to cite our paper by:

```
@article{CDFAG,
  title={Transferable Feature Representation for Visible-to-Infrared Cross-Dataset Human Action  Recognition},
  author={Yang Liu and Zhaoyang Lu and Jing Li and Chao Yao and Yanzi Deng},
  journal={Complexity},
  volume={2018},
  pages = {1--20},
  year={2018},
  doi = {10.1155/2018/5345241}
}
```

### Reference

Yang Liu, Zhaoyang Lu, Jing Li, Chao Yao, and Yanzi Deng, “Transferable Feature Representation for Visible-to-Infrared Cross-Dataset Human Action Recognition,” *Complexity*, vol. 2018, Article ID 5345241, 20 pages, 2018. doi:10.1155/2018/5345241





