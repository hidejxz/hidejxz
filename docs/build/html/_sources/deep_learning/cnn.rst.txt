卷积神经网络(CNN)
========================================

卷积神经网络一般用在图像识别上.一个完整的卷积神经网络通常由卷积层、池化层和全连接层组成.其中全连接层与普通神经网络的连接方式没有太多区别,重点讲一下卷积层与池化层.

卷积层
-------------

卷积的过程就是对图像特征提炼的过程。具体的卷积操作如下图所示:

.. image:: ../images/cnn_1.png
    :width: 800px
    :align: center

左图是一个6 * 6的黑白图片的矩阵,中间是个3 * 3过滤器.将过滤器映射到矩阵的左上角,把对应的数字相乘并求和,得到了等式右边的矩阵的第一个值.如此依次滑动过滤器窗口,每次都做相应的乘积与求和,最终得到目标4 * 4的矩阵.这样的操作就叫卷积.不同的过滤器起到的作用也不同,例如可以检测图片中物品的边缘等.

* Padding

如此的卷积操作,会让图片的尺寸越来越小,并且会丢失边缘位置的信息,这种情况在神经网络层数非常深的时候尤为明显.为了避免这个问题,会在卷积操作时对原图片矩阵的边缘进行填充:

.. image:: ../images/cnn_2.png
    :width: 800px
    :align: center

* Stride

过滤器的步长也是可以设置的:

.. image:: ../images/cnn_3.png
    :width: 800px
    :align: center

* 单层卷积神经网络

一个单层的卷积神经网络如图所示:

.. image:: ../images/cnn_4.png
    :width: 800px
    :align: center

一张彩色的图片通常由RGB三个颜色通道组成.与多个过滤器经过卷积计算并激活后,堆叠成一个新的多层矩阵,作为下一层神经元的输入.一些相关符号与公式如下图所示:

.. image:: ../images/cnn_5.png
    :width: 800px
    :align: center


池化层
-------------

池化层相对来说比较简单,通常用来缩减模型的大小,提高计算速度,同时提高所提取的特征的鲁棒性.

.. image:: ../images/cnn_6.png
    :width: 800px
    :align: center

池化层也是类似卷积层,会有一个过滤窗口,在过滤窗口内取最大值为Max-Pooling,取平均值叫Average-Pooling.池化层只有filter size和stride这2个超参数,没有其他的参数需要训练


经典网络
-------------

神经网络需要训练的参数与超参数量很大.对于一些特定的问题,可以尝试用相关的经典网络取解决,也有一些比较成熟的方案,去优化一些计算上的问题.下面简单列举一些经典的卷积神经网络及方案:

* LeNet

.. image:: ../images/cnn_lenet.png
    :width: 800px
    :align: center

* AlexNet

.. image:: ../images/cnn_alexnet.png
    :width: 800px
    :align: center

* VGG-16

.. image:: ../images/cnn_vgg16.png
    :width: 800px
    :align: center

* ResNet

ResNet残差网络,会在层与层之间增加跳跃连接,以减少计算成本:

.. image:: ../images/cnn_resnet.png
    :width: 800px
    :align: center

* Inception Network

当不确定过滤器用什么大小时,可以采用不同大小的过滤器与池化层堆叠的形式,并通过1 * 1的过滤器减少信道数,来降低计算成本:

.. image:: ../images/cnn_inception.png
    :width: 800px
    :align: center










