# 第三方模型与来源

本项目整合两个 **现成的深度学习 ONNX 推理模型**，项目代码本身未从零训练模型权重。

| 模型 | 用途 | OpenCV 官方仓库 | 模型目录许可 | 官方文件 SHA-256 |
|---|---|---|---|---|
| YuNet (2023mar) | 人脸检测、5 个关键点 | https://huggingface.co/opencv/face_detection_yunet | MIT | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| SFace (2021dec) | 对齐后的深度人脸特征 | https://huggingface.co/opencv/face_recognition_sface | Apache 2.0 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

- OpenCV Zoo: https://github.com/opencv/opencv_zoo
- OpenCV 官方教程: https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
- YuNet 论文: Wu, Peng and Yu, *YuNet: A Tiny Millisecond-level Face Detector*, Machine Intelligence Research, 2023.
- SFace 论文: Zhong et al., *SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition*, IEEE TIP, 2021.

模型代码/权重的第三方许可与归属按上游文档处理，生成的本地特征模板是敏感的生物特征数据。打包源码时不携带任何 OpenCV Zoo ONNX 权重，也不携带个人照片或注册数据库。
