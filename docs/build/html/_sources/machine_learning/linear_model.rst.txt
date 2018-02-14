线性回归与逻辑回归
========================================
线性回归
----------
* 一般式:  

.. math::
    h_\theta(x)=\theta^Tx=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n

* 损失函数(L2正则):  

.. math::
    J(\theta)=\frac{1}{2m}\left[\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^n\theta_j^2\right]

* 目标: 最小化损失函数，即:  

.. math::
    \min\ J(\theta)

* 求解

.. math::
    \theta = (X^TX)^{-1}X^TY

* 梯度下降一般式:  

.. math::
    \theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)

* 线性回归的梯度下降:

.. math::
   \begin{align}
   \theta_j & := \theta_j-\alpha\left[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j\right]\\
   & := \theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
   \end{align}


逻辑回归
----------
* 一般式:  

.. math::
    h_\theta(x)=g(\theta^Tx)

* Sigmoid Function

.. math::
    g(z)=\frac{1}{1+e^{-z}}

.. image:: ../images/lr_sigmoid.png
    :width: 200px
    :align: center

* 损失函数(L2正则):

.. math::
    \begin{align}
    Cost(h_\theta(x),y) & = 
    \begin{cases}
    -\log(h_\theta(x)) & \mbox{if }y=1 \\
    -\log(1-h_\theta(x)) & \mbox{if }y=0
    \end{cases}\\
    & = -y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
    \end{align}

.. math::
    \begin{align}
    J(\theta) & = \frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2\\
    & = -\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
    \end{align}

* 逻辑回归的梯度下降(正好形式与线性回归一致，区别在于 :math:`h_\theta(x)` 不同): 

.. math::
   \begin{align}
   \theta_j & := \theta_j-\alpha\left[\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}\theta_j\right]\\
   & := \theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
   \end{align}

正则化
---------
正则化的目的是防止模型过拟合。常见的正则项有:

.. math::
    \begin{align}
    Lasso(L1正则化) & : \lambda\sum_{j=1}^m\theta_j\\
    Ridge(L2正则化) & : \lambda\sum_{j=1}^m\theta_j^2\\
    Elastic Net & : \lambda\rho\sum_{j=1}^m\theta_j+\frac{\lambda(1-\rho)}{2}\sum_{j=1}^m\theta_j^2
    \end{align}

其中L1正则化还可以进行特征选择:

.. image:: ../images/lr_regularization.jpg
    :width: 500px
    :align: center


