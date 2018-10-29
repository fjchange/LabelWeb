# label_web
Inter-Video Labeled Data-set Processing, to Match the Same Class from Different Video Data-set. 

跨视频数据集整合，希望借助Encoder抽取出图片的特征向量，然后拥有各个图片的特征向量，每一个视频下的每一个类已经标记好，那么就可以利用该类的特征向量计算向量的中心，对每一个视频下每一个类的中心利用距离比较计算，找出最相邻的向量，然后辅助以人工判断，实现跨视频的数据整合。

1. FaceNet为Encoder，训练的时候是采用最大的不相交的5个数据集进行训练，然后对已知为同一栏的视频整合。（整合并非必要项，先实现 90%的训练集，10%的测试集的情形下的训练，获得openset下训练的结果。多次训练，十折交叉验证法）

2. 人脸预训练网络模型来源 https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55- ，在dl-server中应该有

项目架构，各文件内容：
1. facenet.py facenet的包括数据获取，三元组产生，LFW测试

主要需要修改的文件：
1. train_new_model.py 训练的主要文件内容，包括数据的前期处理
2. utils.py 数据的读取操作
3. face_net_test.py 这里是修改后的测试的文件


4. predit.py 用来读取训练后的模型，产生图片的Encode后的向量
5. clustering.py 匹配的交互界面，与产生每一个类的中心
