# Face Related Papers and Code Collection
Any face research/engineer related merge request is wellcome! 02/08/2019.

## Index
1. [Toolkits](#toolkit)
2. [Face Detection](#face-detection)
    - [Survey](#face-detection-survey)
    - [Datasets](#face-detection-datasets)
    - [Research](#face-detection-research)
3. [Face Alignment](#face-alignment)
    - [Survey](#face-alignment-survey)
    - [Datasets](#face-alignment-datasets)
    - [Research](#face-alignment-research)
4. [Face Recosntruction](#face-reconstruction)
    - [Survey](#face-reconstruction-survey)
    - [Datasets](#face-reconstruction-datasets)
    - [Research](#face-reconstruction-research)
5. [Face Recognition](#face-recognition)
    - [Survey](#face-recognition-survey)
    - [Tutorial](#face-recognition-tutorial)
    - [Datasets](#face-recognition-datasets)
    - [Template Generator](#face-recognition-template-generator)
        - [Pretrained models](#face-recognition-pre-trained-model)
        - [Image-based Template Generator](#face-recognition-image-template-generator)
        - [Image-set-based Template Generator](#face-recognition-set-template-generator)
    - [Face Recognition Pipeline](#face-recognition-pipeline)
6. [Face Generation](#face-generation)
    - [Survey](#face-generation-survey)
    - [Datasets](#face-generation-datasets)
    - [Research](#face-generation-research)
7. [Face Attributes Analysis](#face-attributes-analysis)
    - [Survey](#face-attributes-analysis-survey)
    - [Datasets](#face-attributes-analysis-datasets)
    - [Research](#face-attributes-analysis-research)


## Toolkits <a name="toolkit"></a>
- FaRE: Open Source Face Recognition Performance Evaluation Package [[Paper](https://arxiv.org/abs/1901.09447)]  [Code is coming soon!]
- Gluon Toolkit for Face Recognition [[MXNET](https://github.com/THUFutureLab/gluon-face)] 
- Deep Learning:
    - [MXNet](mxnet.io) and [Gluon](http://gluon.mxnet.io/): A flexible and efficient library for deep learning.
    - [Torch](torch.ch) and [PyTorch](pytorch.org): Tensors and Dynamic neural networks in Python with strong GPU acceleration.
    - [TensorFlow](tensorflow.org): An open-source software library for Machine Intelligence.
    - [Caffe](caffe.berkeleyvision.org) and [Caffe2](https://github.com/caffe2/caffe2): A lightweight, modular, and scalable deep learning framework.
- Machine Learning:
    - [Dlib](http://dlib.net/ml.html): A machine learning toolkit.
- Computer Vision:
    - [OpenCV](http://opencv.org/): Open Source Computer Vision Library.
- Probabilistic Programming
    - [Pyro](https://github.com/uber/pyro): Deep universal probabilistic programming with Python and PyTorch

## Face Detection <a name="face-detection"></a>
### Survey <a name="face-detection-survey"></a>

### Datasets <a name="face-detection-datasets"></a>
- [Wildest Faces: Face Detection and Recognition in Violent Settings](https://arxiv.org/abs/1805.07566)
- [WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/paper.pdf) [[Project](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)] 
- [FDDB: Face Detection and Data Set Benchmark](https://www.cics.umass.edu/~elm/papers/fddb.pdf) [[Project](http://vis-www.cs.umass.edu/fddb/)] 
- [AFLW: Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf) [[Project](https://lrs.icg.tugraz.at/research/aflw/)] 

### Research <a name="face-detection-research"></a>
- PyramidBox: A Context-assisted Single Shot Face Detector [ [Paper](https://arxiv.org/pdf/1803.07737.pdf)]  [[TensorFlow](https://github.com/EricZgw/PyramidBox)]  [[PyTorch](https://github.com/Goingqs/PyramidBox)]  [[MXNet](https://github.com/JJXiangJiaoJun/gluon_PyramidBox)]  
- Face Attention Network: An Effective Face Detector for the Occluded Faces [[Paper](https://arxiv.org/abs/1711.07246)]  [[PyTorch](https://github.com/rainofmine/Face_Attention_Network)]  
- FaceNess-Net: Face Detection through Deep Facial Part Responses: [[Paper](https://arxiv.org/pdf/1701.08393.pdf)] 
- S<sup>3</sup>FD: Single Shot Scale-invariant Face Detector [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)]  [[Caffe](https://github.com/sfzhang15/SFD)]  [[PyTorch](https://github.com/clcarwin/SFD_pytorch)] 
- Finding Tiny Faces: [[Project](https://www.cs.cmu.edu/~peiyunh/tiny/)]  [[Paper](https://arxiv.org/abs/1612.04402)]  [[MatConvNet + MATLAB](https://github.com/peiyunh/tiny)]  [[TensorFlow](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)]  [[MXNET](https://github.com/zzw1123/mxnet-finding-tiny-face)] 
- SSH: Single Stage Headless Face Detector: [[Paper](https://arxiv.org/pdf/1708.03979.pdf)]  [[Caffe](https://github.com/mahyarnajibi/SSH)]  [[TensorFlow](https://github.com/DetectionTeamUCAS/SSH_Tensorflow)]  [[MXNET](https://github.com/deepinsight/mxnet-SSH)]  
- Focal Loss for Dense Object Detection: [[Paper](https://arxiv.org/abs/1708.02002)]  [[Caffe](https://github.com/chuanqi305/FocalLoss)]  [[TensorFlow](https://github.com/ailias/Focal-Loss-implement-on-Tensorflow)]  [[MXNET](https://github.com/unsky/focal-loss)] 
- Face R-CNN: [[Paper](https://arxiv.org/abs/1706.01061)]  [[Caffe](https://github.com/playerkk/face-py-faster-rcnn)] 
- FaceBoxes: A CPU Real-time Face Detector with High Accuracy [[Paper](http://cn.arxiv.org/abs/1708.05234)]  [[Caffe](https://github.com/zeusees/FaceBoxes)]  
- Multiview Face Detection: [[Paper](https://arxiv.org/abs/1502.02766)]  [[Caffe](https://github.com/guoyilin/FaceDetection_CNN)] 
    
## Face Alignment <a name="face-alignment"></a>
### Survey <a name="face-alignment"></a>

### Datasets <a name="face-alignment"></a>
- LS3D-W: How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) [[Project](https://www.adrianbulat.com/face-alignment)] 
- AFLW: Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. [[Project](https://lrs.icg.tugraz.at/research/aflw/)] 
- 300-W [[Project](https://ibug.doc.ic.ac.uk/resources/300-W/)] 
- 300-VW [[Project](https://ibug.doc.ic.ac.uk/resources/300-VW/)]

### Research <a name="face-alignment"></a>
- FAN: How far are we from solving the 2D & 3D Face Alignment problem? [[Paper](https://arxiv.org/abs/1703.07332)]  [[PyTorch](https://github.com/1adrianb/face-alignment)] 
- JFA: Joint Head Pose Estimation and Face Alignment Framework
Using Global and Local CNN Features [[Paper](http://cbl.uh.edu/pub_files/07961802.pdf)] 
- MDM: Mnemonic Descent Method [[Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/trigeorgis2016mnemonic.pdf)]  [[TensorFlow](https://github.com/trigeorgis/mdm)] 
- RDL: Recurrent 3D-2D Dual Learning for Large-pose Facial Landmark Detection [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xiao_Recurrent_3D-2D_Dual_ICCV_2017_paper.pdf)] 
- PIFA: Pose-invariant 3D face alignment [[Paper](https://arxiv.org/abs/1506.03799)]  [[Code](http://cvlab.cse.msu.edu/project-pifa.html)] 

## Face Reconstruction <a name="face-reconstruction"></a>
### Survey <a name="face-reconstruction-survey"></a>

### Datasets <a name="face-reconstruction-datasets"></a>

### Research <a name="face-reconstruction-research"></a>
- UH-E2FAR: End-to-end 3D face reconstruction with deep neural networks: [[Paper](https://arxiv.org/abs/1704.05020)] 
- Multi-View 3D Face Reconstruction with Deep Recurrent Neural Networks: [[Paper](http://cbl.uh.edu/pub_files/IJCB-2017-PD.pdf)] 
- 3D Face Morphable Models "In-the-Wild" [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Booth_3D_Face_Morphable_CVPR_2017_paper.pdf)] 
- 3DMM-CNN [[Paper](https://arxiv.org/pdf/1612.04904.pdf)]  [[Code](https://github.com/anhttran/3dmm_cnn)] 
- VRN [[Paper](https://arxiv.org/pdf/1703.07834.pdf)]  [[Code](https://github.com/AaronJackson/vrn)] 
- 3DFaceNet [[Paper](https://arxiv.org/pdf/1708.00980.pdf)] 
- MoFA: Unsupervised learning for 3D model and pose parameters [[Paper](https://arxiv.org/abs/1703.10580)] 
- 3DMM-STN: Using 3DMM to transfer 2D image to 2D image texture [[Paper](https://arxiv.org/abs/1708.07199)] 
- Dense Semantic and Topological Correspondence of 3D Faces without Landmarks
- Generating 3D Faces using Convolutional Mesh Autoencoders [[Paper](https://arxiv.org/pdf/1807.10267.pdf)]  [[Code](https://github.com/anuragranj/coma)] 

## Face Recognition <a name="face-recognition"></a>
### Survey <a name="face-recognition-survey"></a>

### Tutorial <a name="face-recognition-tutorial"></a>
- [Deep Learning for Face Recognition](http://valse.mmcheng.net/deep-learning-for-face-recognition/)

### Datasets <a name="face-recognition-datasets"></a>
#### Training sets:
- MS-Celeb-1M: Microsoft dataset contains around 1M subjects [[Project](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)]  [[Paper](https://arxiv.org/abs/1607.08221)]  
- CASIA WebFace: 10,575 subjects and 494,414 images [[Project](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)]  [[Paper](http://arxiv.org/abs/1411.7923)]  
- CelebA: 202,599 images and 10,177 subjects, 5 landmark locations, 40 binary attributes [[Project](http://mmlab.ie.cuhk.edu.hk/projects/)] 
- VGG-Face2: A large-scale face dataset contains 3.31 million imaes of 9131 identities. [[Project](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)] 

#### Face Verification
- LFW: Labeled Face in the Wild: 13,000 images and 5749 subjects [[Download](http://vis-www.cs.umass.edu/lfw/)]
- CFP: Celebrities in Frontal-Profile in the Wild [[Project](http://www.cfpw.io/)]  [[Paper](http://www.cfpw.io/paper.pdf)]
- MegaFace: 1 Million Faces for Recognition at Scale, 690,572 subjects [[Download](http://megaface.cs.washington.edu/)]
- Surveillance Face Recognition Challenge [[Project](https://qmul-survface.github.io/)]  [[Paper](https://arxiv.org/abs/1804.09691)] 

#### Face Closed-set Identification
- UHDB31: UHDB31: A Dataset for Better Understanding Face Recognition
across Pose and Illumination Variation [[Paper](http://cbl.uh.edu/pub_files/UHDB31_-_CHI_Workshop_-_Final)] 

#### Face Open-set Identification
- IJB-C: IARPA Janus Benchmark-C: Face dataset and protocol [[Paper](https://noblis.org/wp-content/uploads/2018/03/icb2018.pdf)]
- IJB-B: IARPA Janus Benchmark-B Face Dataset [[Paper](https://www.nist.gov/document/ijbbchallengedocumentationreadmepdf)]
- IJB-A: Pushing the frontiers of unconstrained face detection and recognition: IARPA Janus Benchmark A [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1B_089_ext.pdf)]
- Unconstrained Face Detection and Open-Set Face Recognition Challenge [[Project](http://vast.uccs.edu/Opensetface/)]  [[Paper](https://arxiv.org/abs/1708.02337)] 
- MegaFace: 1 Million Faces for Recognition at Scale, 690,572 subjects [[Download](http://megaface.cs.washington.edu/)]

### Template Generators <a name="face-recognition-template-generator"></a>
#### Pretrained Models <a name="face-recognition-pre-trained-model"></a>
- ResNet-101, DenseNet-121 provided by [FaRE](https://arxiv.org/abs/1901.09447)
- ResNet-50,  SE-ResNet-50 provided by [VGG-Face2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) [[Download](https://github.com/ox-vgg/vgg_face2)]  
- VGG-16 provided by [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
- InsightFace [[Download](https://github.com/deepinsight/insightface)]

#### Image-based Template Genearator <a name="face-recognition-image-template-generator"></a>
- Pairwise Relation Network, ECCV18: [[Paper](https://arxiv.org/pdf/1808.04976.pdf)]
- GridFace: Face Rectification via Learning Local Homography Transformation, ECCV18: [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhou_GridFace_Face_Rectification_ECCV_2018_paper.pdf)] 
- Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition, ECCV18: [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohang_Zhan_Consensus-Driven_Propagation_in_ECCV_2018_paper.pdf)] 
- Face Recognition with Contrastive Convolution, ECCV18: [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chunrui_Han_Face_Recognition_with_ECCV_2018_paper.pdf)] 
- FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR15 [[Paper](https://arxiv.org/abs/1503.03832)]  [[TensorFlow](https://github.com/davidsandberg/facenet)] 
- DeepID series, CVPR14: [[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)]  [[DeepID2](http://arxiv.org/abs/1406.4773)]  [[DeepID3](http://arxiv.org/abs/1502.00873)] 
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification, CVPR14: [[Paper](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)]  

#### Image-set-based Template Generator <a name="face-recognition-set-template-generator"></a>
- Dependency-aware Attention Control for Unconstrained Face Recognition with Image Sets, ECCV, 2018 [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaofeng_Liu_Dependency-aware_Attention_Control_ECCV_2018_paper.pdf)]


#### Training Loss <a name="face-recognition-training-loss"></a>
- InsightFace (ArcFace): Additive Angular Margin Loss for Deep Face Recognition [[Paper](https://arxiv.org/abs/1801.07698)]  [[MXNet](https://github.com/deepinsight/insightface)] 
- CosFace: Large Margin Cosine Loss for Deep Face Recognition [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1797.pdf)]  [[TensorFlow](https://github.com/yule-li/CosFace)]  [[MXNet](https://github.com/deepinsight/insightface)] 
- Ring loss: Convex Feature Normalization for Face Recognition [[Paper](https://arxiv.org/abs/1803.00130)]  [[PyTorch](https://github.com/Paralysis/ringloss)] 
- Git Loss for Deep Face Recognition [[Paper](https://arxiv.org/abs/1807.08512)] 
- A-Softmax Loss (SphereFace) [[Paper](https://arxiv.org/abs/1704.08063)]  [[Caffe](https://github.com/wy1iu/sphereface)] (Caffe) 
- Triplet Loss [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)]  [[Torch](https://github.com/cmusatyalab/openface)]  [[TensorFlow](https://github.com/davidsandberg/facenet)] 
- Center Loss [[Paper](http://ydwen.github.io/papers/WenECCV16.pdf)]  [[Caffe + MATLAB](https://github.com/ydwen/caffe-face)]  [[MXNet](https://github.com/pangyupo/mxnet_center_loss)] 
- Range Loss [[Paper](https://arxiv.org/abs/1611.08976)]  [[Caffe](https://github.com/Charrin/RangeLoss-Caffe)]  
- L-Softmax [[Paper](https://arxiv.org/abs/1612.02295)]  [[Caffe](https://github.com/wy1iu/LargeMargin_Softmax_Loss)]  [[MXNet](https://github.com/luoyetx/mx-lsoftmax)] 
- Marginal Loss [[Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/deng_marginal_loss_for_cvpr_2017_paper.pdf)] 

### Face Recognition Pipeline <a name="face-recognition-pipeline"></a>
- UR2D-E:Evaluation of a 3D-aided Pose Invariant 2D Face Recognition System [[Paper](http://cbl.uh.edu/pub_files/IJCB-2017-XX.pdf))] 
- SeetaFaceEngine: An open source C++ face recognition engine. [[C++](https://github.com/seetaface/SeetaFaceEngine)] 
- OpenFace: Face recognition with Google's FaceNet deep neural network using Torch. [[Paper](http://reports-archive.adm.cs.cmu.edu/anon/anon/2016/CMU-CS-16-118.pdf)]  [[Python]((https://github.com/cmusatyalab/openface))] 
 
## Face Genearation <a name="face-generation"></a>
### Survey <a name="face-generation-survey"></a>

### Datasets <a name="face-generation-datasets"></a>

### Research <a name="face-generation-research"></a>
1. TP-GAN: [[Paper](https://arxiv.org/abs/1704.04086)]
2. FF-GAN: [[Paper](https://arxiv.org/abs/1704.06244)]
3. DR-GAN: [[Paper](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)] [[Website](http://cvlab.cse.msu.edu/project-dr-gan.html)]
4. BEGAN: Boundary Equilibrium Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1703.10717)]

## Face Attributes <a name="face-attributes"></a>
### Survey <a name="face-attributes-survey"></a>

### Datasets <a name="face-attributes-datasets"></a>

### Research <a name="face-attributes-research"></a>
