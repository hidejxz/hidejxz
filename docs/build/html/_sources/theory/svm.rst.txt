支持向量机(svm)
========================================
线性
-----------
SVM的线性分类问题得先从逻辑回归说起。

.. image:: ../images/svm_cf.png
    :width: 800px
    :align: center

如上图所示，在y=1和y=0两种情况下，分别对损失函数做了更严格的调整，并对整个损失函数除以了 :math:`\frac{\lambda}{m}` ,即：

.. math:: 
    LR : \min_\theta\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}\left(-\log h_\theta(x^{(i)})\right)+(1-y^{(i)})\left(-\log(1-h_\theta(x^{(i)}))\right)\right]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2

.. math::     
    SVM : \min_\theta C\sum_{i=1}^m\left[y^{(i)}cost_1(\theta^Tx^{(i)})+(1-y)^{(i)}cost_0(\theta^Tx^{(i)})\right]+\frac{1}{2}\sum_{j=1}^n\theta_j^2

为了最小化损失函数，得让函数的第一项无限逼近于0，而正则项最小化，即

.. math::
    \min_\theta\frac{1}{2}\sum_{j=1}^n\theta_j^2

.. math::
    \begin{cases}
    \theta^Tx \ge 1(not\ just \ge 0) & \mbox{if }y=1 \\
    \theta^Tx \leq -1(not\ just \leq 0) & \mbox{if }y=0
    \end{cases}

而

.. math::
    \theta^Tx^{(i)}=p^{(i)}\cdot\|\theta\|

其中 :math:`p` 为 :math:`x` 向量在 :math:`\theta` 向量上的映射，同时也是 :math:`x` 到分界超平面的距离，如下图所示

.. image:: ../images/svm_db.png
    :width: 800px
    :align: center

所以为了最小化目标函数，在 :math:`\theta` 尽可能取最小值的情况下，尽可能增大 :math:`p` 。SVM就是通过这样的方式来增大样本与决策边界的距离，同时最小化特征的权重参数。

非线性
-------------
对于非线性的决策边界，一个简单的典型例子如下图,其中部分特征由高阶项组成:

.. image:: ../images/svm_non_lin.png
    :width: 800px
    :align: center

我们的假设是将所有的样本都映射到另一个空间(高维空间，维度为训练样本的数量m)，而在这个高维空间中样本线性可分。因此:

.. math::
    \theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_1x_2+\theta_4x_1^2+\theta_5x_2^2+\cdots\\
    =\theta_0+\theta_1f_1+\theta_2f_2+\theta_3f_3+\theta_4f_4+\theta_5f_5+\cdots+\theta_mf_m

其中 :math:`f` 是 :math:`x` 到另一个高维空间的映射函数，也称为两者的相似函数，也是传说中的核函数！高斯核函数是一种常用的核函数:

.. math::
    \begin{align}
    f_m & = similarity(x,l^{(i)})=\exp\left(-\frac{\|x-l^{(i)}\|}{2\sigma^2}\right)\\
    & \approx
    \begin{cases}
    1 & \mbox{if }x \approx l^{(i)} \\
    0 & \mbox{if }x\ far\ from\ l^{(i)} 
    \end{cases}
    \end{align}

其中 :math:`l^{(i)}` 即为 :math:`x^{(i)}`

所以总的svm求解步骤可以总结为如下图:

.. image:: ../images/svm_non_lin2.png
    :width: 800px
    :align: center

目标函数:

.. math:: 
    LR : \min_\theta\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}\left(-\log h_\theta(x^{(i)})\right)+(1-y^{(i)})\left(-\log(1-h_\theta(x^{(i)}))\right)\right]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2

.. math:: 
    SVM线性 : \min_\theta C\sum_{i=1}^m\left[y^{(i)}cost_1(\theta^Tx^{(i)})+(1-y)^{(i)}cost_0(\theta^Tx^{(i)})\right]+\frac{1}{2}\sum_{j=1}^n\theta_j^2

.. math:: 
    SVM非线性 : \min_\theta C\sum_{i=1}^m\left[y^{(i)}cost_1(\theta^Tf^{(i)})+(1-y)^{(i)}cost_0(\theta^Tf^{(i)})\right]+\frac{1}{2}\sum_{i=1}^m\theta_i^2


SVM参数:

.. image:: ../images/svm_parameters.png
    :width: 800px
    :align: center

逻辑回归与SVM对比
----------------------

.. image:: ../images/svm_lr_vs_svm.png
    :width: 800px
    :align: center

使用SVM时，样本量要适中。太大会提升计算成本，太小则无法达到低维映射到高维的目的。
