朴素贝叶斯
========================================
* 贝叶斯定理:

.. math::
    P(y|x_1,\cdots,x_n) = \frac{P(y)P(x_1,\cdots,x_n|y)}{P(x_1,\cdots,x_n)}

* 条件独立性假设("朴素"):

.. math::
    P(x_1,\cdots,x_n|y)=\prod_{i=1}^n P(x_i|y) 

* 朴素贝叶斯一般式:

.. math::
    P(y|x_1,\cdots,x_n) = \frac{P(y)\prod_{i=1}^n P(x_i|y)}{P(x_1,\cdots,x_n)}

由上式可得

.. math::
    P(y|x_1,\cdots,x_n) \propto P(y)\prod_{i=1}^n P(x_i|y)

因此朴素贝叶斯法即求

.. math::
    \arg \max_yP(y)\prod_{i=1}^n P(x_i|y)

* 连续特征属性使用高斯模型

.. math::
    P(x_i|y)=\frac{1}{\sqrt{2\pi\sigma_y^2}}\exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right)

* 离散特征属性使用多项式模型

.. math::
    P(x_i|y)=\frac{N_{yi}+\alpha}{N_y+\alpha n}

其中 :math:`N_y` 为 :math:`y` 等于特定值的个数; :math:`N_{yi}` 为 :math:`y` 等于特定值条件下，:math:`x` 等于某值的数量; :math:`n` 为该特征值的种类数。为防止未出现过的特征值对计算概率造成影响，在计算特征值概率时，引入了 :math:`\alpha` 因子，当 :math:`\alpha=1` 时，称为拉普拉斯平滑。

