# *SphereFace*: Deep Hypersphere Embedding for Face Recognition

By Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj and Le Song

### License

SphereFace is released under the MIT License (refer to the LICENSE file for details).

### Update
- **2022.4.10**: **If you are looking for an easy-to-use and well-performing PyTorch implementation of SphereFace, we now have it! Check out our official SphereFace PyTorch re-implementation [here](https://opensphere.world/).**
- **2018.8.14**: We recommend an interesting ECCV 2018 paper that comprehensively evaluates SphereFace (A-Softmax) on current widely used face datasets and their proposed noise-controlled IMDb-Face dataset. Interested users can try to train SphereFace on their IMDb-Face dataset. Take a look [here](https://arxiv.org/pdf/1807.11649.pdf).
- **2018.5.23**: A new *SphereFace+* that explicitly enhances the inter-class separability has been introduced in our technical report. Check it out [here](https://arxiv.org/abs/1805.09298). Code is released [here](https://github.com/wy1iu/sphereface-plus).
- **2018.2.1**: As requested, the prototxt files for SphereFace-64 are released.
- **2018.1.27**: We updated the appendix of our SphereFace paper with useful experiments and analysis. Take a look [here](http://wyliu.com/papers/LiuCVPR17v3.pdf). The content contains:
	- The intuition of removing the last ReLU;
	- Why do we want to normalize the weights other than because we need more geometric interpretation?
	- Empirical experiment of zeroing out the biases;
	- More 2D visualization of A-Softmax loss on MNIST;
	- **Angular Fisher score** for evaluating the angular feature discriminativeness, which is a new and straightforward evaluation metric other than the final accuracy.
	- Experiments of SphereFace on MegaFace with different convolutional layers;
	- The annealing optimization strategy for A-Softmax loss;
	-  Details of the 3-patch ensemble strategy in MegaFace challenge;

- **2018.1.20**: We updated some resources to summarize the current advances in angular margin learning. Take a look [here](#resources-for-angular-margin-learning).

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Installation](#installation)
0. [Usage](#usage)
0. [Models](#models)
0. [Results](#results)
0. [Video Demo](#video-demo)
0. [Note](#note)
0. [Third-party re-implementation](#third-party-re-implementation)
0. [Resources for angular margin learning](#resources-for-angular-margin-learning)


### Introduction

The repository contains the entire pipeline (including all the preprocessings) for deep face recognition with **`SphereFace`**. The recognition pipeline contains three major steps: face detection, face alignment and face recognition.

SphereFace is a recently proposed face recognition method. It was initially described in an [arXiv technical report](https://arxiv.org/abs/1704.08063) and then published in [CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf). The most up-to-date paper with more experiments can be found at [arXiv](https://arxiv.org/abs/1704.08063) or [here](http://wyliu.com/papers/LiuCVPR17v3.pdf). To facilitate the face recognition research, we give an example of training on [CAISA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and testing on [LFW](http://vis-www.cs.umass.edu/lfw/) using the **20-layer CNN architecture** described in the paper (i.e. SphereFace-20). 

In SphereFace, our network architectures use residual units as building blocks, but are quite different from the standrad ResNets  (e.g., BatchNorm is not used, the prelu replaces the relu, different initializations, etc). We proposed 4-layer, 20-layer, 36-layer and 64-layer architectures for face recognition (details can be found in the [paper]((https://arxiv.org/pdf/1704.08063.pdf)) and [prototxt files](https://github.com/wy1iu/sphereface/blob/master/train/code/sphereface_model.prototxt)). We provided the 20-layer architecture as an example here. If our proposed architectures also help your research, please consider to cite our paper.

SphereFace achieves the state-of-the-art verification performance (previously No.1) in [MegaFace Challenge](http://megaface.cs.washington.edu/results/facescrub.html#3) under the small training set protocol.


### Citation

If you find **SphereFace** useful in your research, please consider to cite:

	@InProceedings{Liu_2017_CVPR,
	  title = {SphereFace: Deep Hypersphere Embedding for Face Recognition},
	  author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Li, Ming and Raj, Bhiksha and Song, Le},
	  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year = {2017}
	}

Our another closely-related previous work in ICML'16 ([more](https://github.com/wy1iu/LargeMargin_Softmax_Loss)):

	@InProceedings{Liu_2016_ICML,
	  title = {Large-Margin Softmax Loss for Convolutional Neural Networks},
	  author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Yang, Meng},
	  booktitle = {Proceedings of The 33rd International Conference on Machine Learning},
	  year = {2016}
	}


### Requirements
1. Requirements for `Matlab`
2. Requirements for `Caffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
3. Requirements for `MTCNN` (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)) and `Pdollar toolbox` (see: [Piotr's Image & Video Matlab Toolbox](https://github.com/pdollar/toolbox)).

### Installation
1. Clone the SphereFace repository. We'll call the directory that you cloned SphereFace as **`SPHEREFACE_ROOT`**.

    ```Shell
    git clone --recursive https://github.com/wy1iu/sphereface.git
    ```

2. Build Caffe and matcaffe

    ```Shell
    cd $SPHEREFACE_ROOT/tools/caffe-sphereface
    # Now follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html
    make all -j8 && make matcaffe
    ```

### Usage

*After successfully completing the [installation](#installation)*, you are ready to run all the following experiments.

#### Part 1: Preprocessing
**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_ROOT/preprocess/`**
1. Download the training set (`CASIA-WebFace`) and test set (`LFW`) and place them in **`data/`**.

	```Shell
	mv /your_path/CASIA_WebFace  data/
	./code/get_lfw.sh
	tar xvf data/lfw.tgz -C data/
	```
    Please make sure that the directory of **`data/`** contains two datasets.
    
2. Detect faces and facial landmarks in CAISA-WebFace and LFW datasets using `MTCNN` (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)).

	```Matlab
	# In Matlab Command Window
	run code/face_detect_demo.m
	```
    This will create a file `dataList.mat` in the directory of **`result/`**.
3. Align faces to a canonical pose using similarity transformation.

	```Matlab
	# In Matlab Command Window
  	run code/face_align_demo.m
  	```
    This will create two folders (**`CASIA-WebFace-112X96/`** and **`lfw-112X96/`**) in the directory of **`result/`**, containing the aligned face images.

#### Part 2: Train
**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_ROOT/train/`**

1. Get a list of training images and labels.

	```Shell&Matlab
	mv ../preprocess/result/CASIA-WebFace-112X96 data/
	# In Matlab Command Window
	run code/get_list.m
	```
    The aligned face images in folder **`CASIA-WebFace-112X96/`** are moved from ***preprocess*** folder to ***train*** folder. A list `CASIA-WebFace-112X96.txt` is created in the directory of **`data/`** for the subsequent training.

2. Train the sphereface model.

	```Shell
	./code/sphereface_train.sh 0,1
	```
    After training, a model `sphereface_model_iter_28000.caffemodel` and a corresponding log file `sphereface_train.log` are placed in the directory of `result/sphereface/`.

#### Part 3: Test
**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_ROOT/test/`**

1. Get the pair list of LFW ([view 2](http://vis-www.cs.umass.edu/lfw/#views)).

	```Shell
	mv ../preprocess/result/lfw-112X96 data/
	./code/get_pairs.sh
	```
	Make sure that the LFW dataset and`pairs.txt` in the directory of **`data/`**

1. Extract deep features and test on LFW.

	```Matlab
	# In Matlab Command Window
	run code/evaluation.m
	```
    Finally we have the `sphereface_model.caffemodel`, extracted features `pairs.mat` in folder **`result/`**, and accuracy on LFW like this:

	fold|1|2|3|4|5|6|7|8|9|10|AVE
	:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
	ACC|99.33%|99.17%|98.83%|99.50%|99.17%|99.83%|99.17%|98.83%|99.83%|99.33%|99.30%

### Models
1. Visualizations of network architecture (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):
	- SphereFace-20: [link](http://ethereon.github.io/netscope/#/gist/20f6ddf70a35dec5019a539a502bccc5)
2. Model file
	- SphereFace-20: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegb2F6dmlmOXhWaVk) | [Baidu](http://pan.baidu.com/s/1qY5FTF2)
	- Third-party SphereFace-4 & SphereFace-6: [here](https://github.com/wy1iu/sphereface/issues/81) by [zuoqing1988](https://github.com/zuoqing1988)


### Results
1. Following the instruction, we go through the entire pipeline for 5 times. The accuracies on LFW are shown below. Generally, we report the average but we release the [model-3](#models) here.

	Experiment |#1|#2|#3 (released)|#4|#5
	:---:|:---:|:---:|:---:|:---:|:---:
	ACC|99.24%|99.20%|**99.30%**|99.27%|99.13%

2. Other intermediate results:
    - LFW features: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegenU0cGJYZmlRUlU) | [Baidu](http://pan.baidu.com/s/1o8QIMUY)
    - Training log: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegcWkxdVV4X1FOaFU) | [Baidu](http://pan.baidu.com/s/1i5QmXrJ)



### Video Demo
[![SphereFace Demo](https://img.youtube.com/vi/P6jEzzwoYWs/0.jpg)](https://www.youtube.com/watch?v=P6jEzzwoYWs)

Please click the image to watch the Youtube video. For Youku users, click [here](http://t.cn/RCZ0w1c).

Details:
1. It is an **open-set** face recognition scenario. The video is processed frame by frame, following the same pipeline in this repository.
2. Gallery set consists of 6 identities. Each main character has only 1 gallery face image. All the detected faces are included in probe set.
3. There is no overlap between gallery set and training set (CASIA-WebFace).
4. The scores between each probe face and gallery set are computed by cosine similarity. If the maximal score of a probe face is smaller than a pre-defined threshold, the probe face would be considered as an outlier.
5. Main characters are labeled by boxes with different colors. (
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)Rachel,
![#ffff00](https://placehold.it/15/ffff00/000000?text=+)Monica,
![#ff80ff](https://placehold.it/15/ff80ff/000000?text=+)Phoebe,
![#00ffff](https://placehold.it/15/00ffff/000000?text=+)Joey,
![#0000ff](https://placehold.it/15/0000ff/000000?text=+)Chandler,
![#00ff00](https://placehold.it/15/00ff00/000000?text=+)Ross)


### Note
1. **Backward gradient**
	- In this implementation, we did not strictly follow the equations in paper. Instead, we normalize the scale of gradient. It can be interpreted as a varying strategy for learning rate to help converge more stably. Similar idea and intuition also appear in [normalized gradients](https://arxiv.org/pdf/1707.04822.pdf) and [projected gradient descent](https://www.stats.ox.ac.uk/~lienart/blog_opti_pgd.html).
	- More specifically, if the original gradient of ***f*** w.r.t ***x*** can be written as **df/dx = coeff_w \*  w + coeff_x \* x**, we use the normalized version **[df/dx] = (coeff_w \* w + coeff_x \* x) / norm_wx** to perform backward propragation, where **norm_wx** is **sqrt(coeff_w^2 + coeff_x^2)**. The same operation is also applied to the gradient of ***f*** w.r.t ***w***.
	- In fact, you do not necessarily need to use the original gradient, since the original gradient sometimes is not an optimal design. One important criterion for modifying the backprop gradient is that the new "gradient" (strictly speaking, it is not a gradient anymore) need to make the objective value decrease stably and consistently. (In terms of some failure cases for gradient-based back-prop, I recommand [a great talk](https://www.youtube.com/watch?v=jWVZnkTfB3c) by [Shai Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/))
	- If you use the original gradient to do the backprop, you could still make it work but may need different lambda settings, iteration number and learning rate decay strategy. 

2. **Lambda** and **Note for training (When the loss becomes 87)**
	- Please refer to our previous [note and explanation](https://github.com/wy1iu/LargeMargin_Softmax_Loss#notes-for-training).
	
3. **According to recent advances, using feature normalization with a tunable scaling parameter s can significantly improve the performance of SphereFace on MegaFace challenge**
	- This is supported by the experiments done by [CosFace](https://arxiv.org/abs/1801.09414). Similar idea also appears in [additive margin softmax](https://arxiv.org/abs/1801.05599).

4. **Difficulties in convergence**
        - When you encounter difficulties in convergence (it may appear if you use *SphereFace* in another dataset), usually there are a few easy ways to address it.
	- First, try to use large mini-batch size. 
	- Second, try to use PReLU instead of ReLU. 
	- Third, increase the width and depth of our network. 
	- Fourth, try to use better initialization. For example, use the pretrained model from the original softmax loss (it is also equivalent to finetuning).
	- Last and the most effective thing you could try is to change the hyper-parameters for lambda_min, lambda and its decay speed.


### Third-party re-implementation
- PyTorch: [code](https://github.com/clcarwin/sphereface_pytorch) by [clcarwin](https://github.com/clcarwin).
- PyTorch: [code](https://github.com/Joyako/SphereFace-pytorch) by [Joyako](https://github.com/Joyako).
- TensorFlow: [code](https://github.com/pppoe/tensorflow-sphereface-asoftmax) by [pppoe](https://github.com/pppoe).
- TensorFlow (with awesome animations): [code](https://github.com/YunYang1994/SphereFace) by [YunYang1994](https://github.com/YunYang1994).
- TensorFlow: [code](https://github.com/hujun100/tensorflow-sphereface) by [hujun100](https://github.com/hujun100).
- TensorFlow: [code](https://github.com/HiKapok/tf.extra_losses) by [HiKapok](https://github.com/HiKapok).
- TensorFlow: [code](https://github.com/andrewhuman/sphereloss_tensorflow) by [andrewhuman](https://github.com/andrewhuman).
- MXNet: [code](https://github.com/deepinsight/insightface) by [deepinsight](https://github.com/deepinsight) (by setting loss-type=1: SphereFace).
- Model compression for SphereFace: [code](https://github.com/isthatyoung/Sphereface-prune) by [Siyang Liu](https://github.com/isthatyoung) (useful in practice)
- Caffe2: [code](https://github.com/tpys/face-recognition-caffe2) by [tpys](https://github.com/tpys).
- Trained on MS-1M: [code](https://github.com/KaleidoZhouYN/Sphereface-Ms-celeb-1M) by [KaleidoZhouYN](https://github.com/KaleidoZhouYN).
- System: [A cool face demo system](https://github.com/tpys/face-everthing) using SphereFace by [tpys](https://github.com/tpys).
- Third-party pretrained models: [code](https://github.com/goodluckcwl/Sphereface-model) by [goodluckcwl](https://github.com/goodluckcwl).

### Resources for angular margin learning

[L-Softmax loss](https://github.com/wy1iu/LargeMargin_Softmax_Loss) and [SphereFace](https://github.com/wy1iu/sphereface) present a promising framework for angular representation learning, which is shown very effective in deep face recognition. We are super excited that our works has inspired many well-performing methods (and loss functions). We list a few of them for your potential reference (not very up-to-date):

- Additive margin softmax: [paper](https://arxiv.org/abs/1801.05599) and [code](https://github.com/happynear/AMSoftmax)
- CosFace: [paper](https://arxiv.org/abs/1801.09414)
- ArcFace/InsightFace: [paper](https://arxiv.org/abs/1801.07698) and [code](https://github.com/deepinsight/insightface)
- NormFace: [paper](https://arxiv.org/abs/1704.06369) and [code](https://github.com/happynear/NormFace)
- L2-Softmax: [paper](https://arxiv.org/abs/1703.09507)
- von Mises-Fisher Mixture Model: [paper](https://arxiv.org/abs/1706.04264)
- COCO loss: [paper](https://arxiv.org/pdf/1710.00870.pdf) and [code](https://github.com/sciencefans/coco_loss)
- Angular Triplet Loss: [code](https://github.com/KaleidoZhouYN/Angular-Triplet-Loss)

To evaluate the effectiveness of the angular margin learning method, you may consider to use the **angular Fisher score** proposed in [the Appendix E of our SphereFace Paper](http://wyliu.com/papers/LiuCVPR17v3.pdf).

Disclaimer: Some of these methods may not necessarily be inspired by us, but we still list them due to its relevance and excellence.

### Contact

  [Weiyang Liu](https://wyliu.com) and [Yandong Wen](https://ydwen.github.io)

  Questions can also be left as issues in the repository. We will be happy to answer them.
