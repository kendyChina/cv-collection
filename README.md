# CNN之思考  
## 频域之于CNN泛化性的解释 2020  
**论文：** [https://arxiv.org/abs/1905.13545](https://arxiv.org/abs/1905.13545)  
**标题：** High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks  
**作者：** Haohan Wang, Xindi Wu, Zeyi Huang, Eric P. Xing  
School of Computer Science  
Carnegie Mellon University  
**收录：** CVPR 2020  
**代码：** [https://github.com/HaohanWang/HFC](https://github.com/HaohanWang/HFC)  
  
**关键词：**  
  
 - 图像可以划分为低频信息（语义信息）和高频信息（边缘或突变），人在标注label时只基于低频信息；  
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210103222837572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
 - 在高低频信息分离后，CNN可能会对低频图像进行错误预测，对高频图像进行正确预测。证明CNN训练时可能对两者共同学习；  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210103223839635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
 - 对于同一类别数据，在不同场景下，语义分布应该近乎一致，但高频分布则可能与特定域有关。**故CNN的泛化能力与数据分布特性有关**；  
 - CNN训练时，会先拟合低频信息，在loss值达到瓶颈时，会进一步尝试拟合高频信息；  
![左图为正确标注的训练数据，右图为被打乱标注的训练数据。L(r)越大则高频保留越多。左图在训练初期有更高的accu，证明初期对低频信息的依赖。两者后期都能达到相近的accu，但右图所需epoch更多。](https://img-blog.csdnimg.cn/20210103224128766.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
 - **高频信息包含与数据分布特性相关的成分，及与数据无关的噪声。** 但CNN无法针对性的训练，故如果训练时噪声引入过多则会出现过拟合，导致泛化能力下降；  
 - **早停**预防过拟合的能力，可能在于预防模型利用高频信息中的噪声进行训练；  
 - 大batch size训练时，会同时考虑更多的非噪声的同分布高频成分，从而缩小train和test的gap；  
 - mix-up缩小了train和test的gap，是因为混淆了低频信息，从而鼓励CNN尽可能多去捕获高频信息；  
 - 加入对抗样本会快速降低CNN精度。因为对抗样本可能改变了高频分布（人眼无法感知），train阶段实际学到的高频分布和对抗样本的高频分布不一致，故预测错误；  
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021010322494031.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
 - Vanilla指无BN的模型。BN优势之一是通过归一化来对齐不同信号的分布差异，让模型更容易获取高频成分。且由于对齐效应，大部分捕获的是有益高频成分，从而加快收敛、提高泛化能力；  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210103225634413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
 - 暂无较好方式去除高频信息中的噪声，目前唯一能尝试用合适的半径阈值 r 去除部分高频信息，预防噪声干扰。同时test也要进行相应高频过滤，或许能提高泛化能力；  
 - 对抗鲁棒性较好的模型卷积核更加平滑，可以利用该特性稍微提高CNN的鲁棒性。  
  
参考文档：[https://mp.weixin.qq.com/s/jdy07LGbChMuSuGjX5qOJg](https://mp.weixin.qq.com/s/jdy07LGbChMuSuGjX5qOJg)  
  
---  
  
# 目标检测  
## R-CNN  
## Fast R-CNN  
## Faster R-CNN  
## YOLO  
## YOLOv2  
## YOLOv3  
## YOLOv4 2020  
**YOLOv4**：**Y**ou **O**nly **L**ook **O**nce v4  
**论文：** [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)  
**标题：** YOLOv4: Optimal Speed and Accuracy of Object Detection  
**作者：** Alexey Bochkovskiy∗  
  
Chien-Yao Wang∗  
Institute of Information Science  
Academia Sinica, Taiwan  
  
Hong-Yuan Mark Liao  
Institute of Information Science  
Academia Sinica, Taiwan  
**代码：** [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)  
  
**关键词：**  
  
 - 对多种网络结构、训练技巧的组合进行了充分实验，对在图像分类和目标检测的作用做了对比；  
 - 作者目标是让使用者可以轻松使用GPU训练和测试实时的、高质量的结果，在Darknet框架中（如上代码链接）可以轻松使用；  
 - 对目标检测模型的架构进行划分，含Input、Backbone、Neck、Head，结构见下截图；  
 - Bag-of-Freebies（BoF），不增加推理损耗而提高模型精度的技巧。文中列举了：  
	 - 数据增强；  
	 - 分类loss；  
	 - 回归loss。  
 - Bag-of-Spacials（BoS），增加少量推理损耗而提高精度的技巧。文中列举了：  
	 - 提高感受野；  
	 - 引入注意力机制；  
	 - 提高特征聚合能力；  
	 - 激活函数；  
	 - 后处理方法。  
 - 目标检测的模型需要：  
	 - 更高的输入分辨率，以检测小目标；  
	 - 更大的感受野，即更多的层，以cover更高的分辨率；  
	 - 更多的参数，以适配单一图片下不同分辨率的目标。  
 - 引入了新的数据增强方式：Mosaic、Self-Adversarial Training (SAT)：  
	 - Mosaic混合四张image作为一张image，提高了图片的variance，对BN更有利；  
	 - SAT包含两个前向、后向过程。第一阶段，并不更新网络权重，而是修改原始图像，在图像中产生原本没有的物体，从而达到对自身的对抗攻击。在第二阶段，以正常的方式用修改后的图像进行训练。  
 - 提出了修改版SAM、修改版PAN、Cross mini-Batch Normalization (CmBN)  
	 - CmBN和BN相比，BN每个mini-batch只计算当前的Mean和Variance，CmBN会计算当前batch的；  
	 - 而和CBN相比，CBN无论batch，都会计算前4个mini-batch的Mean和Variance，并且每个mini-batch都会更新Bias、Scale、W，CBN则聚焦当前batch的Mean和Variance，且每个batch更新一次Bias、Scale、W（见下图解）；  
	 - 修改版SAM从空间维度的注意力机制改为像素维度的注意力机制（见下图解）；  
	 - 修改版PAN从addition改为connection（见下图解）。  
 - YOLOv4结构上：  
	 - backbone选择CSPDarknet53，因拥有更大的感受野及更低的时延；  
	 - SPP模块及PANet路径聚合neck；  
	 - 延续YOLOv3的head结构。  
 - backbone的BoF：CutMix、Mosaic数据增强、DropBlock正则化、Class label smoothing；  
 - backbone的BoS：Mish激活函数、Cross-stage partial connections (CSP)、Multiinput weighted residual connections (MiWRC)；  
 - detector的BoF：CIoU loss、CmBN、DropBlock、Mosaic、SAT、Eliminate grid sensitivity、多anchor对应单目标、余弦退火策略（ Cosine annealing scheduler）、最优超参数（ Optimal hyperparameters）、随机训练图像尺寸（Random training shapes）；  
 - detector的BoS：Mish、SPP、SAM、PAN、DIoU-NMS；  
 - 作者对各种各种BoF和BoS进行了横向对比、消融实验，在落地到自己项目时，可以作为参考。  
  
![目标检测模型结构](https://img-blog.csdnimg.cn/20210110105726691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
![目标检测模型常见模块](https://img-blog.csdnimg.cn/20210110110223678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
![Mosaic数据增强方式示例](https://img-blog.csdnimg.cn/20210110115714620.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
![BN CBN CmBN图解](https://img-blog.csdnimg.cn/20210110120605498.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
![修改版SAM图解](https://img-blog.csdnimg.cn/20210110120645392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
![修改版PAN图解](https://img-blog.csdnimg.cn/20210110120706646.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
  
  
  
---  
  
# 小目标检测  
---  
## COCO小目标检测的数据增强 2019  
**论文：** [https://arxiv.org/abs/1902.07296](https://arxiv.org/abs/1902.07296)  
**标题：** Augmentation for small object detection  
**作者：** Mate Kisantal, Zbigniew Wojna, Jakub Murawski, Jacek Naruniec, Kyunghyun Cho  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210109110913675.png)  
**关键词：**  
  
 - 针对小目标图片较少的问题，使用过采样OverSampling的策略，  
 - 针对一张图片中小目标较少的问题，在图片中用分割的Mask（COCO数据集有分割Mask）进行裁剪复制，添加旋转、缩放等变换后，在不遮挡其余目标的前提下粘贴；  
策略2能让Anchor-base模型有更多的正样本  
 - 以上两种策略会对其他尺寸（尤其大尺寸）样本造成负面影响，故要衡量两者的重要性；  
 - 作者对以上两种策略作了各种对比，包括：过采样比例、过采样与粘贴的组合方式、小目标粘贴的数量、对几种（一种、几种、全部）小目标粘贴、粘贴是否重叠以及是否边缘模糊。  
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021010911100919.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210109111136570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70)  
  
  
  
---  
  
  
# 图像分割  
## FCN  
---  
## U-Net  
---  
## DeepLab  
  
---  
  
# 轻量型算法  
  
## MobileNet  
## ShuffleNet  
## SqueezeNet  
## MixNet  
## GhostNet  
  
  
---  
  
  
# 目标跟踪  
---  
  
## 数据集  
---  
  
### VOT  
**VOT：** **V**isual **O**bject **T**racking  
**官网：** [https://www.votchallenge.net/](https://www.votchallenge.net/)  
**评价指标(2020)：**  
  
 - Accuracy  
 准确率  
 - Robustness  
 鲁棒性，也是错误率  
 - EAO(**E**xpected **A**verage **O**verlap)  
综合了**Accuracy**和**Robustness**的评价指标  
  
评价指标每年都不太一样，可参考官方对当年Challenge的公告。  
  
---  
  
### OTB  
  
  **OTB：** **O**bject **T**racking **B**enchmark  
  官网：[http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)  
  
  
  
---  
  
  
## GOTURN 2016  
**GOTURN**：**G**eneric **O**bject **T**racking **U**sing **R**egression **N**etworks  
**官网：** [https://davheld.github.io/GOTURN/GOTURN.html](https://davheld.github.io/GOTURN/GOTURN.html)  
**论文：** [https://davheld.github.io/GOTURN/GOTURN.pdf](https://davheld.github.io/GOTURN/GOTURN.pdf)  
**标题：** Learning to Track at 100 FPS with Deep  
Regression Networks  
**作者：** David Held, Sebastian Thrun, Silvio Savarese  
Department of Computer Science  
Stanford University  
{davheld,thrun,ssilvio}@cs.stanford.edu  
**收录：** European Conference on Computer Vision (ECCV), 2016 (In press)  
**代码：** [https://github.com/davheld/GOTURN](https://github.com/davheld/GOTURN)  
  
**关键词：**  
  
 - 离线学习  
 - 单目标跟踪  
 - **Regression任务**：根据上一frame的crop region，在当前frame的search region中regress Bounding Box  
 - 根据目标的移动速度和fps，可适当调节crop size  
 - 除了基于video训练，还基于形变后的image训练，以扩大训练集  
 - test时需初始化Bounding Box  
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118113856634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118113920479.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118114007825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118114019526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
  
---  
  
## MDNet 2015  
**MDNet：** **M**ulti-**D**omain Convolutional Neural **Net**work Tracker  
**官网：** [http://cvlab.postech.ac.kr/research/mdnet/](http://cvlab.postech.ac.kr/research/mdnet/)  
**论文：** [https://arxiv.org/abs/1510.07945](https://arxiv.org/abs/1510.07945)  
**标题：** Learning Multi-Domain Convolutional Neural Networks for Visual Tracking  
**作者：** Hyeonseob Nam Bohyung Han  
Dept. of Computer Science and Engineering, POSTECH, Korea  
{namhs09, bhhan}@postech.ac.kr  
**竞赛：** The Winner of **The VOT2015 Challenge**  
**代码：** [https://github.com/HyeonseobNam/MDNet](https://github.com/HyeonseobNam/MDNet)  
  
**关键词：**  
  
 - 单目标跟踪，需初始化Bounding Box  
 - 由CNN，Fully Connection组成  
 - **二分类任务**：fc6为二分类器，判断候选框为目标或背景  
 - **离线学习**进行预训练，**在线学习**进行fine-tune  
 - **离线学习**：相同backbone（CNN+fc4+fc5），以及k个fc6，由k个视频训练，以学习通用的特征层(backbone)，和特定的分类层(fc6)  
 - **在线学习**：新建fc6，frozen CNN部分，在线fine-tune fc4-fc6  
 - **在线学习**结合**long-term updates**和**short-term updates**策略进行更新  
 - **Hard Minibatch Mining**：参考**hard negative mining**采样策略组合batch  
 - 仅在在线学习第一帧进行**regressor**训练，降低损耗  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118183602672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118183613807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201118183645320.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
  
---  
  
## ROLO 2016  
**ROLO：** **R**ecurrent Y**OLO** for object tracking  
**官网：** [http://guanghan.info/projects/ROLO/](http://guanghan.info/projects/ROLO/)  
**论文：** [https://arxiv.org/abs/1607.05781](https://arxiv.org/abs/1607.05781)  
**标题：** Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking  
**作者：** Guanghan Ning∗  
, Zhi Zhang, Chen Huang, Zhihai He  
Department of Electrical and Computer Engineering  
University of Missouri  
Columbia, MO 65201  
{gnxr9, zzbhf, chenhuang, hezhi}@mail.missouri.edu  
Xiaobo Ren, Haohong Wang  
TCL Research America  
{renxiaobo, haohong.wang}@tcl.com  
**代码：** [https://github.com/Guanghan/ROLO](https://github.com/Guanghan/ROLO)  
**翻译：** [https://blog.csdn.net/MacKendy/article/details/110040151](https://blog.csdn.net/MacKendy/article/details/110040151)  
  
**关键词：**  
  
 - YOLO + LSTM  
 - 单目标跟踪，无需初始化Bounding Box，离线学习  
 - Regression任务  
 - 按原文所述：效果比传统跟踪算法（如YOLO+卡尔曼滤波）要好，尤其在特殊场合鲁棒性更强  
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201124170048828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201124170057196.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hY0tlbmR5,size_16,color_FFFFFF,t_70#pic_center)  
  
---  
  
## SORT 2016  
**SORT：** **S**imple **O**nline and **R**ealtime **T**racking  
**官网：**  
**论文：** [https://arxiv.org/abs/1602.00763](https://arxiv.org/abs/1602.00763)  
**标题：** Simple Online and Realtime Tracking  
**作者：** Alex Bewley†, Zongyuan Ge†, Lionel Ott⋄, Fabio Ramos⋄, Ben Upcroft†  
Queensland University of Technology†, University of Sydney⋄  
**代码：** [https://github.com/abewley/sort](https://github.com/abewley/sort)  
**翻译：** [https://blog.csdn.net/MacKendy/article/details/110118796](https://blog.csdn.net/MacKendy/article/details/110118796)  
  
**关键词：**  
  
 - 在线跟踪，离线学习，多目标跟踪，无需初始化Bounding Box  
 - 能适应视频中增加或减少目标  
 - Detection + Kalman Filter + Hungarian algorithm  
 - 跟踪性能取决于Detector性能，可根据跟踪任务选择合适的Detector及对应训练集  
 - Kalman Filter跟踪目标Bounding Box的[u, v, s, r]，并更新[u, v, s]，分别是目标中心的坐标、尺寸、纵横比  
 - 基于IoU距离计算cost矩阵，通过Hungarian算法assign目标  
  
---  
  
## DeepSORT 2017  
  
**DeepSORT：**  
**官网：**  
**论文：** [https://arxiv.org/abs/1703.07402](https://arxiv.org/abs/1703.07402)  
**标题：**  
**作者：**  
**代码：** [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort)  
**翻译：**  
