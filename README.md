## Auto-Encoder and VAE
### 1. AUto-Encoder
![](https://pic2.zhimg.com/80/v2-e5745659cd57562c1dcfc3de7e2a4229_720w.png)

### 2. VAE
#### 1. 模型介绍
![](https://pic4.zhimg.com/80/v2-67f83e248d59359e4833e8219bad4a77_720w.jpg)
![](https://spaces.ac.cn/usr/uploads/2018/03/4168876662.png)
#### 2. KL损失函数
![](./pic/KL.png)

### 3. 模型比较
下图从左至右分别为**原始图像**，**MLP_AE生成的图像**，**Conv_AE生成的图像**，**VAE生成的图像**

从下图我们可以看出**Conv_AE生成的图像**，**VAE生成的图像** 优于 **MLP_AE生成的图像**

![](./pic/orig.png) ![](./pic/mlp_ae.png) ![](./pic/conv_ae.png) ![](./pic/vae.png)

### 4. 参考
- [花式解释AutoEncoder与VAE](https://zhuanlan.zhihu.com/p/27549418)
- [VAE中的损失函数-impact of the loss function](https://zhuanlan.zhihu.com/p/345360992)
- [变分自编码器（一）：原来是这么一回事](https://spaces.ac.cn/archives/5253)