# *SphereFace* : Deep Hypersphere Embedding for Face Recognition

By Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj and Le Song

### Introduction

The repository contains the entire pipeline (including all the preprossings) for deep face recognition with **`SphereFace`**. The recognition pipeline contains three major steps: face detection, face alignment and face recognition.

SphereFace is a recently proposed face recognition method. It was initially described in an [arXiv technical report](https://arxiv.org/pdf/1704.08063.pdf) and then published in [CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf). To facilitate the face recognition research, we give an example of training on [CAISA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and testing on [LFW](http://vis-www.cs.umass.edu/lfw/) using the **20-layer CNN architecture** described in the paper (i.e. SphereFace-20). 

SphereFace achieves one of the top verification performance (previously No.1) in [MegaFace Challenge](http://megaface.cs.washington.edu/results/facescrub.html#3) under the small training set protocol.

### License

SphereFace is released under the MIT License (refer to the LICENSE file for details).

### Citing SphereFace

If you find **SphereFace** useful in your research, please consider to cite:

    @inproceedings{liu2017sphereface,
        author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Li, Ming and Raj, Bhiksha and Song, Le},
        title = {SphereFace: Deep Hypersphere Embedding for Face Recognition},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        Year = {2017}
    }

Our another closely-related previous work in ICML'16 ([more](https://github.com/wy1iu/LargeMargin_Softmax_Loss)):

    @inproceedings{liu2016large,
        author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Yang, Meng},
        title = {Large-Margin Softmax Loss for Convolutional Neural Networks},
        booktitle = {Proceedings of The 33rd International Conference on Machine Learning},
        Year = {2016}
    }

### Video Demo
[![SphereFace Demo](https://img.youtube.com/vi/P6jEzzwoYWs/0.jpg)](https://www.youtube.com/watch?v=P6jEzzwoYWs)

Please click the image to watch the Youtube video. For Youku users, click [here](http://t.cn/RCZ0w1c).

Details:
1. It is an **open-set** face recognition scenario. The video is processed frame by frame, following the same pipeline in this repository.
2. Gallery set consists of 6 identities. Each main character has only 1 gallery face image. All the detected faces are included in probe set.
3. There is no overlap between gallery set and training set (CASIA-WebFace).
4. The scores between each probe face and gallery set are computed by cosine similarity. If the maximal score of a probe face is smaller than a pre-definded threshold, the probe face would be considered as an outlier.
5. Main characters are labeled by boxes with different colors. (
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)Rachel,
![#ffff00](https://placehold.it/15/ffff00/000000?text=+)Monica,
![#ff80ff](https://placehold.it/15/ff80ff/000000?text=+)Phoebe,
![#00ffff](https://placehold.it/15/00ffff/000000?text=+)Joey,
![#0000ff](https://placehold.it/15/0000ff/000000?text=+)Chandler,
![#00ff00](https://placehold.it/15/00ff00/000000?text=+)Ross)



### Contents
1. [Update](#update)
2. [Note](#note)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Models](#models)
7. [Results](#results)

### Update
- August 16, 2017
  * A video demo is released.
  * Training log and extracted features are released.
- August 9, 2017
  * Most of the bugs are fixed. 
  * SphereFace-20 prototxt file (**`$SPHEREFACE_ROOT/train/code/sphereface_model.prototxt`**) is released. This architecture is exactly the same as the 20-layer CNN reported in the paper. 
  * A well-trained model with accuracy **99.30%** on **LFW** is released.
- July 20, 2017
  * This repository was built.

### Note
1. **Backward gradient.**
	- In this implementation, we did not strictly follow the equations in paper. Instead, we normalize the scale of gradient. It can be interpreted as a varying strategy for learning rate to help converge more stably. Similar idea and intuition also appear in [normalized gradients](https://arxiv.org/pdf/1707.04822.pdf) and [projected gradient descent](https://www.stats.ox.ac.uk/~lienart/blog_opti_pgd.html).
	- More specifically, if the original gradient of ***f*** w.r.t ***x*** can be written as **df/dx = coeff_w \*  w + coeff_x \* x**, we use the normalized version **[df/dx] = (coeff_w \* w + coeff_x \* x) / norm_wx** to perform backward propragation, where **norm_wx** is **sqrt(coeff_w^2 + coeff_x^2)**. The same operation is also applied to the gradient of ***f*** w.r.t ***w***.
	- In fact, you do not necessarily need to use the original gradient, since the original gradient sometimes is not an optimal design. One important criterion for modifying the backprop gradient is that the new "gradient" (strictly speaking, it is not a gradient anymore) need to make the objective value decrease stably and consistently. (In terms of some failure cases for gradient-based back-prop, I recommand [a great talk](https://www.youtube.com/watch?v=jWVZnkTfB3c) by [Shai Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/))
	- If you use the original gradient to do the backprop, you could still make it work but may need different lambda settings, iteration number and learning rate decay strategy. 

2. **Lambda.**
	- Please refer to our previous [explanation](https://github.com/wy1iu/LargeMargin_Softmax_Loss#notes) (notes for training).


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
	./code/sphereface/sphereface_train.sh 0,1
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
	- SphereFace-20: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegb2F6dmlmOXhWaVk) | [Baidu](http://pan.baidu.com/s/1skLAPtz)
	
### Results
1. Following the instruction, we go through the entire pipeline for 5 times. The accuracies on LFW are shown below. Generally, we report the average but we release the [model-3](#models) here.

	Experiment |#1|#2|#3 (released)|#4|#5
	:---:|:---:|:---:|:---:|:---:|:---:
	ACC|99.24%|99.20%|**99.30%**|99.27%|99.13%

2. Other intermediate results:
    - LFW features: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegenU0cGJYZmlRUlU) | [Baidu](http://pan.baidu.com/s/1eSgpYuA)
    - Training log: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegcWkxdVV4X1FOaFU) | [Baidu](http://pan.baidu.com/s/1kV7QeUz)



### Contact

  [Weiyang Liu](https://wyliu.com) and [Yandong Wen](https://ydwen.github.io)

  Questions can also be left as issues in the repository. We will be happy to answer them.
