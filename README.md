# Face related papers collection
A collection for the face-related papers and codes.

Updated on 11/04/2017

## Face Detection
1. Multiview Face Detection: [[Paper](https://arxiv.org/abs/1502.02766)]  [[Code](https://github.com/guoyilin/FaceDetection_CNN)] (Caffe + Python)
2. Finding Tiny Faces: [[Project](https://www.cs.cmu.edu/~peiyunh/tiny/)][[Paper](https://arxiv.org/abs/1612.04402)] [[Code](https://github.com/peiyunh/tiny)] (MatConvNet + MATLAB)
3. Focal Loss: [[Paper](https://arxiv.org/abs/1708.02002)]
4. SSH: [[Paper](https://arxiv.org/pdf/1708.03979.pdf)]
5. Face R-CNN: [[Paper](https://arxiv.org/abs/1706.01061)]
6. FaceBoxes: [[Paper](http://cn.arxiv.org/abs/1708.05234)] [[Code](https://github.com/zeusees/FaceBoxes)] (Caffe)
7. S<sup>3</sup>FD: [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)]

## Face Alignment
In this part, you can retrieve the paper before 2016 on this [site](https://sites.google.com/site/yanghengcv/face-alignment).
1. PIFA: Pose-invariant 3D face alignment [[Paper](https://arxiv.org/abs/1506.03799)] [[Code](http://cvlab.cse.msu.edu/project-pifa.html)]
2. MDM: Mnemonic Descent Method [[Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/trigeorgis2016mnemonic.pdf)] [[Code](https://github.com/trigeorgis/mdm)]
3. JFA: Joint head pose estimation and face alignment [[Paper](http://cbl.uh.edu/pub_files/07961802.pdf)]
4. GoDP: Globally optimized dual path way [[Paper](https://arxiv.org/abs/1704.02402)]
5. Recurrent 3D-2D Dual Learning: [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xiao_Recurrent_3D-2D_Dual_ICCV_2017_paper.pdf)

## Face Reconstruction
1. UH-E2FAR [[Paper](https://arxiv.org/abs/1704.05020)]
2. Multi-View RNN [[Paper](http://cbl.uh.edu/pub_files/IJCB-2017-PD.pdf)]
3. 3D Face Morphable Models "In-the-Wild" [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Booth_3D_Face_Morphable_CVPR_2017_paper.pdf)]
4. 3DMM-CNN [[Paper](https://arxiv.org/pdf/1612.04904.pdf)] [[Code](https://github.com/anhttran/3dmm_cnn)]
5. VRN [[Paper](https://arxiv.org/pdf/1703.07834.pdf)] [[Code](https://github.com/AaronJackson/vrn)]
6. 3DFaceNet [[Paper](https://arxiv.org/pdf/1708.00980.pdf)]
7. MoFA: Unsupervised learning for 3D model and pose parameters [[Paper](https://arxiv.org/abs/1703.10580)]
8. 3DMM-STN: Using 3DMM to transfer 2D image to 2D image texture [[Paper](https://arxiv.org/abs/1708.07199)]

## Face GAN
1. TP-GAN: [[Paper](https://arxiv.org/abs/1704.04086)]
2. FF-GAN: [[Paper](https://arxiv.org/abs/1704.06244)]
3. DR-GAN: [[Paper](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)] [[Website](http://cvlab.cse.msu.edu/project-dr-gan.html)]
4. BEGAN: Boundary Equilibrium Generative Adversarial Networks [[Paper](https://arxiv.org/abs/1703.10717)]

## Feature Representation
### Network
1. DeepFace: [[Paper](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)]
2. DeepID series: [[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)] [[DeepID2](http://arxiv.org/abs/1406.4773)] [[DeepID3](http://arxiv.org/abs/1502.00873)]
3. [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/): VGG-Face CNN descriptor.
4. [VGG-Face2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/): VGG-Face ResNet descriptor.

### Loss
1. Triplet Loss [[Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)][[Code](https://github.com/cmusatyalab/openface)](Torch) [[Code](https://github.com/davidsandberg/facenet)](TensorFlow)
2. Center Loss [[Paper](http://ydwen.github.io/papers/WenECCV16.pdf)] [[Code](https://github.com/ydwen/caffe-face)](Caffe + MATLAB) [[Code](https://github.com/pangyupo/mxnet_center_loss)] (MxNet)
3. Range Loss [[Paper](https://arxiv.org/abs/1611.08976)] [[Code](https://github.com/Charrin/RangeLoss-Caffe)] (Caffe)
4. L-Softmax [[Paper](https://arxiv.org/abs/1612.02295)] [[Code](https://github.com/wy1iu/LargeMargin_Softmax_Loss)] (Caffe) [[Code](https://github.com/luoyetx/mx-lsoftmax)] (MxNet)
5. A-Softmax Loss (SphereFace) [[Paper](https://arxiv.org/abs/1704.08063)] [[Code](https://github.com/wy1iu/sphereface)] (Caffe)
6. Marginal Loss [[Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/deng_marginal_loss_for_cvpr_2017_paper.pdf)]

## Pipelines
1. [OpenFace](https://github.com/cmusatyalab/openface): Face recognition with Google's FaceNet deep neural network using Torch.
2. [SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine): An open source C++ face recognition engine.
3. [UR2D](http://cbl.uh.edu/pub_files/IJCB-2017-XX.pdf): 3D-aided 2D Face Recognition system

## Tutorial
1. [Deep Learning for Face Recognition](http://valse.mmcheng.net/deep-learning-for-face-recognition/)

## Datasets
1. Detection
    1. FDDB: Face Detection and Data Set Benchmark. 5k images [[Download](http://vis-www.cs.umass.edu/fddb/)]
    2. AFLW: Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization. 25K images. [[Download](https://lrs.icg.tugraz.at/research/aflw/)]
2. Recognition:
    1. MS-Celeb-1M: Microsoft dataset contains around 1M subjects [[Paper](https://arxiv.org/abs/1607.08221)] [[Download](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)]
    2. CASIA WebFace: 10,575 subjects and 494,414 images [[Paper](http://arxiv.org/abs/1411.7923)] [[Download](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)]
    3. LFW: 13,000 images and 5749 subjects [[Download](http://vis-www.cs.umass.edu/lfw/)]
    4. CelebA: 202,599 images and 10,177 subjects, 5 landmark locations, 40 binary attributes [[Download](http://mmlab.ie.cuhk.edu.hk/projects/)]
    5. MegaFace: 1 Million Faces for Recognition at Scale, 690,572 subjects [[Download](http://megaface.cs.washington.edu/)]
    6. VGG-Face2: A large-scale face dataset contains 3.31 million imaes of 9131 identities. [[Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)]

## Libraries
1. Deep Learning:
    1. [MXNet](mxnet.io) and [Gluon](http://gluon.mxnet.io/): A flexible and efficient library for deep learning.
    2. [Torch](torch.ch) and [PyTorch](pytorch.org): Tensors and Dynamic neural networks in Python with strong GPU acceleration.
    3. [TensorFlow](tensorflow.org): An open-source software library for Machine Intelligence.
    4. [Caffe](caffe.berkeleyvision.org) and [Caffe2](https://github.com/caffe2/caffe2): A lightweight, modular, and scalable deep learning framework.
2. Machine Learning:
    1. [Dlib](http://dlib.net/ml.html): A machine learning toolkit.
3. Computer Vision:
    1. [OpenCV](http://opencv.org/): Open Source Computer Vision Library.
4. Probabilistic Programming
    1. [Pyro](https://github.com/uber/pyro): Deep universal probabilistic programming with Python and PyTorch

