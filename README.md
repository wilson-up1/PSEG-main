# PSEG-main
Here is the code implementation of PSEG. We will present our model and model weights here, as well as the PSEG implementation for adding each model!  



## data

You can download our dataset through Baidu Netdisk.

[DATA](https://pan.baidu.com/s/1qGxpiE3a-vXQ9bs0-2QX9g?pwd=g3l4)

[GT](https://pan.baidu.com/s/1EfMShHkGD0_zR44d41PLNw?pwd=qp2p)

These datasets are all 2D human edge datasets. Our team has checked them to ensure accuracy.

You can extract this file and place it in./data in the root directory.



## model

You can download our dataset through [Baidu Netdisk](https://pan.baidu.com/s/1gBGf8STuWfub1aV6pNbhvw?pwd=r8sl).

The folder includes pre-trained weights, model files, example instantiation files, and configuration files



If you have downloaded the files we provided, please arrange them according to this file directory

```
${ROOT_DIRECTORY}
├── data
├── models
	├──pretrain.pth
	├──model.py
	├──edge.py
	├──utils.py
├── README.md
└── requirements.txt
```





## How to work？

Among the methods we offer, the one we provide is the simplest model fusion method. But please believe that if you use our method, you will definitely achieve an improvement beyond your expectations.

```
Algorithm 1 PSEG-Based Input Enhancement
Input: Raw RGB image X_input∈R^(H×W×3)
Output: Enhanced image X_final∈R^(H×W×3)
Require: Pretrained PSEG model M, 1×1 convolution layer C
1:	Y_gray  = M(X_input) 	▷ PSEG inference
·	Y_gray∈R^(H×W×1), X_input∈R^(H×W×3) 
2:	Y_processing= Dilate(Erode(Y_gray)) 	▷Opening operation
3:	X_concat= X_input⊕ Y_processing     ▷ Channel-wise concatenation
·	X_concat∈R^(H×W×(3+1)) 
4:	X_final=C(X_concat)         ▷ 1×1 convolution with 3 output channels
5:	Return X_final	▷ Immediately following the original model

```



## Other applications

If you are looking for a suitable network and want to try our project, we recommend that you try basic networks such as CorwdPose, HRNet, and TransPose, which can be easily found in open-source projects



## Contribution

Welcome to submit issues and Pull requests!



## Citation of the thesis

If you have used this project in your research, please cite our paper (to be released soon) or Github.
