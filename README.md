# *SphereFace* : Deep Hypersphere Embedding for Face Recognition

By Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj and Le Song

SphereFace was initially described in an [arXiv technical report](https://arxiv.org/abs/1704.08063) and was then published in [CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf).

### Introduction

The repository contains the entire pipeline (including all the preprossings) for deep face recognition with SphereFace. The recognition pipeline contains three major steps: Face detection, face alignment and face recognition. To facilitate all the reseachers, we specify all these three steps in the repository. SphereFace is our proposed face recognition method. For face detection, we use the [MT-CNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment). 

The provided network prototxt example is a 28-layer CNN, which is the same as [Center Face](https://github.com/ydwen/caffe-face). To fully reproduce the results in the paper, you need to make some modifications (e.g. network architecture) according to the SphereFace paper.

### License

SphereFace is released under the MIT License (refer to the LICENSE file for details).

### Citing SphereFace

If you find SphereFace useful in your research, please consider to cite:

    @inproceedings{liu2017sphereface,
        author = {Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song},
        title = {SphereFace: Deep Hypersphere Embedding for Face Recognition},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        Year = {2017}
    }

### Contents
1. [Update](#update)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)

### Update
- July 20, 2017
  * This repository was built.
- To be update: pretrained models, some intermediate results and extracted features will be released soon.

### Requirements
1. Requirements for `Matlab`
2. Requirements for `Caffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
3. Requirements for `MTCNN` (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)) and `Pdollar toolbox` (see: [Piotr's Image & Video Matlab Toolbox](https://github.com/pdollar/toolbox)).

### Installation
1. Clone the SphereFace repository. We'll call the directory that you cloned SphereFace into `SPHEREFACE_ROOT`

    ```Shell
    git clone --recursive https://github.com/wy1iu/sphereface.git
    ```

2. Build Caffe and matcaffe

    ```Shell
    cd $SPHEREFACE_ROOT/tools/caffe-sphereface
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html
    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make matcaffe
    ```

### Usage

*After successfully completing [installation](#installation)*, you'll be ready to run all the following experiments.

#### Part 1: Preprocessing
**Note 1:** In this part, we assume you are in the directory *$SPHEREFACE_ROOT/preprocess/*
1. Download the training set (`CASIA-WebFace`) and test set (`LFW`) and place them in $SPHEREFACE_ROOT/preprocess/data/.

	```Shell
	cd $SPHEREFACE_ROOT/preprocess
	mv /your_path/CASIA_WebFace  data/
	./code/get_lfw.sh
	tar xvf data/lfw.tgz
	```

2. Detect faces and facial landmarks in CAISA-WebFace and LFW datasets using `MTCNN` (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)).

	```Matlab
	# In Matlab Command Window
	run code/face_detect_demo.m
	```

3. Align faces to a canonical pose using similarity transformation.

	```Matlab
	# In Matlab Command Window
  	run code/face_align_demo.m
  	```

#### Part 2: Train
**Note 2:** In this part, we assume you are in the directory *$SPHEREFACE_ROOT/train/*

1. Get a list of training images.

	```Shell&Matlab
	mv ../preprocess/result/CASIA-WebFace-112X96 data/
	# In Matlab Command Window
	run code/get_list.m
	```

2. Train sphereface model.

	```Shell
	./code/sphereface/sphereface_train.sh 0,1
	```

#### Part 3: Test
**Note 3:** In this part, we assume you are in the directory *$SPHEREFACE_ROOT/test/*

1. Extract deep features and test on LFW.

	```Shell&Matlab
	mv ../preprocess/result/lfw-112X96 data/
	# In Matlab Command Window
	run code/evaluation.m
	```

### Contact

  [Yandong Wen](https://ydwen.github.io) and [Weiyang Liu](https://wyliu.com)

  Questions can also be left as issues in the repository. We will be happy to answer them.
