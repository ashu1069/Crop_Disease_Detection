# Crop_Disease_Detection

The project is aimed to detect diseases in the vulnerable crops of Indian agriculture including Potato, Cluster Bean, Cowpea, Okra. The methodology encompasses computer vision and deep learning techniques with a focus on using different architectures such as VGG16, DenseNet, ResNet, AlexNet, InceptionV3, and Xception. The results were tested on a number of different hyperparameters and concluded with a mean f-1 score of **0.99** on Potato dataset. Notably, all the architectures have been trained on the crop datasets via transfer as well as scratch training method.

## Dataset

| Crop | Lab images | Field images | Total |
| ---- | ---------- | ------------ | ------ |
| Potato | 2152 | 1151 | 3303 |
| Okra | 915 | 588 | 1503 |
| Cowpea | 882 | 638 | 1520 |
| Cluster bean | 758 | 695 | 1453 |

## Results

| Crop | VGG16 | DenseNet | AlexNet | InceptionV3 | Custom CNN |
| ----- | ------------ | -------- | ------ | ------- | -------- |
| Potato | 0.86 | 0.99 | 0.96 | 0.98 | 0.97 |
| Okra | 0.87 | 0.987 | 0.94 | 0.965 | 0.93 |
| Cluster bean | 0.84 | 0.95 | 0.96 | 0.965 | 0.94 |
| Cowpea | 0.845 | 0.964 | 0.958 | 0.97 | 0.956 |

All the above results have been on scratch training method
