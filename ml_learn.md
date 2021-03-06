# 机器学习

####机器学习可以这样来理解:
我们有n个样本(sample)的数据集, 想要预测未知数据的属性,
如果每个样本的数字不只有一个, 比如一个多维的条目(多变量数据
(multivariate data)), 那么样本就有多个属性或特征
####将学习问题分为以下几类
1. 有监督学习(Supervised Learning), 是指数据中包括了我们想预测的属性

* 分类(classification), 样本属于两个或多个类别, 我们希望通过已经标识类别的数据学习来预测未标记的分类
* 回归(regression), 指希望的输出是一个或者多个连续的变量
2. 无监督学习(Unsupervised Learning), 训练数据包括输入向量X的集合, 但没有相对应的目标变量,
这类问题的目标可以是发掘数据中相似样本的分组,为聚类(Clustering),
也可以是确定输入样本空间中的数据分布,为密度估计(density estimation)
还可以是将数据从高维空间投射到两维或三维空间, 便宜进行数据可视化

训练集和测试集 机器学习是关于如何从数据学习到一些属性并且用于新的数据集,
这也是为什么机器学习中评估算法的一个习惯做法是将手头上
已有的数据分成两个部分,
一个部分我们称为 训练集(training set), 用来学习数据的属性
另一部分称为测试集(testing set), 用来测试这些属性

1. 学习和预测(learn&&predict)
2. 模型持久性(Model persistence)


1. 机器学习
2. 分类

* 打造模型,将数据分类到不同的类别
* 这些模型的打造方式,是输入一个数据,其中有预先标记的类别,供算法学习
* 在模型输入类别未经标记的数据,让数据基于训练数据库来预测新数据类别
* 分类是'监督学习'的一种形式
* kNN算法(k-NearestNeighbor)k近邻分类算法
* 从训练样本中选择k个与测试样本'距离'最近的样本,这k个样本中出现频率最高的类别即作为测试样本的类别

3. 回归

* 回归和分类紧密联系在一起的
* 分类是预测离散的类别,回归适用的情况是由连续的数字组成,线性回归就是回归技术的一个例子
* 线性回归(Linear Rgression有监督的学习),给定数据集,学习出线性函数,然后测试这个函数好不好,挑出最好的函数
* 梯度下降(Grandient Descent), 确定cost function最好的值
* Normal Equation也可以确定cost function值

4. 聚类

* 聚类是用来分析未标记过的类别
* 聚类是一种'无监督学习'
* 聚类不是通过案例进行学习,而是通过观察进行学习
* K-Means聚类

5. 关联

* 关联属于'无监督学习'
* Aprori关联算法
* 购物篮分析,假设购物者在购物篮中放入各种各样物品,目标是去识别其关联,并分配置信度(是统计学概念,某个样本在样本总体参数的区间评估)

6. 决策树

* 决策树是一种自上而下,分布解决的递归分析器
* 决策树通常由两种任务组成:归纳和修剪
* 归纳是用一组预先分类的数据作为输入,
* 决策树是一种有监督的学习

7. 支持向量机(SVM)(support vector machine)

* SVM可以分为分类线性与非线性数据
* SVM的原理是将训练数据转化进入更高的纬度,
* 再检查这个纬度中的最优间隔距离,或者不同分类中的边界
* 在SVM中,这些边界被称为'超平面'
* svm是一种分类算法
* 有监督的学习


8. 神经网络

* 神经网络是以人类大脑为灵感的算法
* 神经网络是由无数相互连接的概念化人工神经元组成,
* 这些神经元相互之间传送数据,有不同的相关权重,
* 这些权重基于神经网络的'经验'而定,
* 神经元有激活阀值,如果各个神经元权重的结合达到阀值,
* 神经元就是激发,神经元的激发的结合就会带来'学习'

9. 深度学习

* 深度学习是应用深度神经网络技术--具有多个隐藏神经元层的神经网络架构--来解决问题
* 深度学习是一个过程,正如使用了神经网络架构的数据挖掘,这是一种独特的机器学习算法

10. 增强学习

* 增强学习是在某一种情景中寻找最合适的行为,从而最大化奖励

11. K层交叉检验

* 交叉校验是一种打造模型的方法
* 通过去除数据库k层中的一层,训练所有k-1层的数据,
* 用剩下的第k层进行测验
* 然后,这个过程重复k次
* 每一次使用不同层中的测试数据,将错误结果在一个整合模型中结合和平均起来
* 这样做的目的是生成最精准的预测模型

12. 贝叶斯

* 贝叶斯是一种预测概率的机器学习算法
* 贝叶斯分类算法
