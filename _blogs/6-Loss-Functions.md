---
title: "6\\. 常见损失函数的推导：从 MLE 到 MSE / MAE / CE"
date: 2026-04-28
---
Derivation of common loss functions from a unified MLE / NLL perspective: MSE, MAE, CE, and beyond.

把"为什么是这个 loss"说清楚，需要回到一个统一框架：**假设数据由某个概率模型生成，写出似然，取负对数，得到 loss**。Mean Absolute Error (MAE) / Mean Squared Error (MSE) / Cross Entropy (CE) 都是这个框架在不同噪声假设下的特例。这篇 blog 顺着这条主线把三种最常见的回归 / 分类损失推一遍，再延伸到正则化、Focal、Hinge、InfoNCE 等几个常见变体。

---

## 1) 统一视角：MLE → Loss

设观测数据 $\{(x_i, y_i)\}_{i=1}^N$ 是 i.i.d. 采样，假设条件分布 $y_i \mid x_i \sim p_\theta(y \mid x_i)$。最大似然估计 (MLE) 写成：

$$
\hat\theta
=
\arg\max_\theta \prod_i p_\theta(y_i \mid x_i)
=
\arg\min_\theta \Bigl[-\sum_i \log p_\theta(y_i \mid x_i)\Bigr]
$$

定义负对数似然（NLL）：

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_i \log p_\theta(y_i\mid x_i)
$$

这就是统一框架——**所有"逐样本的 NLL"都是一种 loss**。下面三种 loss 都从这里出发。

为什么要取 $\log$？三个互相强化的理由：

* **数值稳定**：似然是 $N$ 个 $(0,1]$ 概率的乘积，直接乘容易下溢；取 $\log$ 把乘法变加法。
* **优化友好**：负对数似然把"乘积最大化"变成"求和最小化"，每个样本贡献一个独立可加的项，方便 mini-batch 随机梯度下降 (Stochastic Gradient Descent, SGD)。
* **信息论解释**：$-\log p(y)$ 就是事件 $y$ 的信息量；NLL 就是平均信息量，连接到熵和 Kullback-Leibler (KL) 散度（参见 5 号 blog [《KL Divergence：从熵到 Forward/Reverse KL》](./5-KL-Divergence.md)）。

NLL 也等价于一种经验风险最小化 (Empirical Risk Minimization, ERM)：把"用 $-\log p_\theta$ 这个 loss 函数衡量预测好坏"当作经验风险来最小化。

> 不同的分布假设 → 不同的 NLL → 不同的 loss。MSE / MAE / CE 的差别本质上就是分布假设的差别。

---

## 2) MSE：高斯噪声假设

### 2.1 推导

设 $y_i = f_\theta(x_i) + \epsilon_i,\quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$，即

$$
p_\theta(y_i \mid x_i)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\Bigl(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\Bigr)
$$

取负对数：

$$
-\log p_\theta(y_i\mid x_i)
=
\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}
+
\underbrace{\log(\sqrt{2\pi}\sigma)}_{\text{与 }\theta\text{ 无关}}
$$

求和并丢掉常数，再吸收 $1/(2\sigma^2)$ 进学习率：

$$
\boxed{\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N}\sum_i (y_i - f_\theta(x_i))^2}
$$

### 2.2 为什么 MSE 估的是均值

很多人记得"MSE 对应均值"，但少有人记得为什么。这其实是一个独立的小事实：

**命题**：对于任意随机变量 $Y$，用一个常数$c$来预测$Y$，$\arg\min_c \mathbb{E}[(Y-c)^2] = \mathbb{E}[Y]$。

证明只是把方差展开：

$$
\mathbb{E}[(Y-c)^2]
=
\mathbb{E}[Y^2] - 2c\,\mathbb{E}[Y] + c^2
$$

对 $c$ 求导令其为 0：$-2\mathbb{E}[Y] + 2c = 0 \Rightarrow c = \mathbb{E}[Y]$。

把这个事实用到条件期望上：固定 $x$，最优常数是 $\mathbb{E}[Y \mid x]$。所以**最小化 MSE 等价于让 $f_\theta(x)$ 逼近条件均值**——这也是为什么 MSE 对离群点敏感：均值会被尾部样本拉走。

### 2.3 性质

* **梯度**：$\nabla_{f}\mathcal{L} = \frac{2}{N}(f - y)$，与残差成线性。
* **凸性**：作为关于 $f$ 的二次函数，处处严格凸；线性模型时有闭式解（normal equation）。
* **对离群点敏感**：残差被平方放大，一个 $|y - f| = 10$ 的离群点贡献是 $|y-f|=1$ 普通点的 100 倍。

---

## 3) MAE：拉普拉斯噪声假设

### 3.1 推导

设 $\epsilon_i \sim \mathrm{Laplace}(0, b)$，密度为

$$
p_\theta(y_i\mid x_i)
=
\frac{1}{2b}\exp\!\Bigl(-\frac{|y_i - f_\theta(x_i)|}{b}\Bigr)
$$

取负对数并丢常数：

$$
\boxed{\mathcal{L}_{\mathrm{MAE}} = \frac{1}{N}\sum_i |y_i - f_\theta(x_i)|}
$$

### 3.2 为什么 MAE 估的是中位数

**命题**：对于任意随机变量 $Y$，$\arg\min_c \mathbb{E}[|Y-c|]$ 是 $Y$ 的中位数。

简短证明：把 $\mathbb{E}[|Y-c|]$ 拆成 $c$ 左右两侧的积分，

$$
\mathbb{E}[|Y-c|]
=
\int_{-\infty}^{c}(c-y)\,p(y)\,dy
+
\int_{c}^{\infty}(y-c)\,p(y)\,dy
$$

对 $c$ 求导（用 Leibniz 法则）：

$$
\frac{d}{dc}\mathbb{E}[|Y-c|]
=
\Pr(Y \le c) - \Pr(Y > c)
$$

令其为 0，要求 $\Pr(Y \le c) = \Pr(Y > c) = 1/2$，即 $c$ 是中位数。

中位数对极端值不敏感（一个离群点最多把中位数挪一个名次），这就是 MAE 对离群点鲁棒的根源。

### 3.3 性质与缺点

* **梯度**：$\partial \mathcal{L}/\partial f = \mathrm{sign}(f - y)$，模长恒为 1（$f = y$ 处不可导，需要次梯度）。
* **鲁棒**：对应中位数。
* **难精细收敛**：梯度恒定 → 接近最优时步长不缩，容易在最优解附近"震荡"。实践中常配合学习率衰减或换 Huber loss。

### 3.4 Huber loss：MSE / MAE 的折中

$$
\mathcal{L}_\delta(r)
=
\begin{cases}
\tfrac{1}{2}r^2 & |r| \le \delta \\
\delta(|r| - \tfrac{1}{2}\delta) & |r| > \delta
\end{cases}
$$

小残差用 MSE（梯度好），大残差用 MAE（鲁棒）。$\delta$ 控制切换点，是 outlier-detection 的隐式阈值。

---

## 4) CE：Categorical / Bernoulli 假设

### 4.1 二分类：Binary Cross-Entropy (BCE)

$y_i \in \{0,1\}$，模型输出 $\hat p_i = \sigma(z_i) \in (0,1)$，假设 $y_i \sim \mathrm{Bernoulli}(\hat p_i)$：

$$
p(y_i\mid x_i) = \hat p_i^{y_i}(1-\hat p_i)^{1-y_i}
$$

取负对数：

$$
\boxed{\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{N}\sum_i \bigl[y_i\log \hat p_i + (1-y_i)\log(1-\hat p_i)\bigr]}
$$

### 4.2 多分类（CE）

$y_i \in \{1,\dots,K\}$，$\hat p_i = \mathrm{softmax}(z_i)$，假设 $y_i \sim \mathrm{Categorical}(\hat p_i)$：

$$
p(y_i\mid x_i) = \prod_{k=1}^{K} \hat p_{i,k}^{\mathbb{1}[y_i=k]}
$$

取负对数：

$$
\boxed{\mathcal{L}_{\mathrm{CE}} = -\frac{1}{N}\sum_i \log \hat p_{i,y_i} = -\frac{1}{N}\sum_i\sum_k \mathbb{1}[y_i=k]\log\hat p_{i,k}}
$$

### 4.3 信息论等价形式（连接 KL）

记真实标签的 one-hot 分布为 $p_i$，模型分布为 $\hat p_i$，则

$$
\mathcal{L}_{\mathrm{CE}}
=
\frac{1}{N}\sum_i H(p_i, \hat p_i)
=
\frac{1}{N}\sum_i \bigl[H(p_i) + D_{\mathrm{KL}}(p_i \,\|\, \hat p_i)\bigr]
$$

由于 $H(p_i)$（one-hot 时为 0，软标签时为常数）与 $\theta$ 无关：

> **最小化 CE = 最小化 $D_{\mathrm{KL}}(p_{\text{data}} \| p_\theta)$，也就是 Forward KL**。

这就把 CE 和 5 号 blog 里的 Forward KL "mode-covering" 直觉对上了：CE 训练之所以鼓励模型在真实标签处给出高概率、不能漏掉任何真实数据出现过的类别，本质就是 Forward KL 的特性。

### 4.4 CE + softmax 的关键梯度推导

经常被当作"已知结论"使用的事实：

$$
\frac{\partial \mathcal{L}_{\mathrm{CE}}}{\partial z_k} = \hat p_k - \mathbb{1}[y=k]
$$

为什么？逐步推一遍。设单样本 loss $L = -\log \hat p_y$，其中 $\hat p_k = e^{z_k}/\sum_j e^{z_j}$。先求 softmax 的 Jacobian：

$$
\frac{\partial \hat p_y}{\partial z_k}
=
\hat p_y(\delta_{ky} - \hat p_k)
$$

（对 $k = y$ 是 $\hat p_y(1-\hat p_y)$，对 $k \ne y$ 是 $-\hat p_y \hat p_k$。）

代入 $L = -\log \hat p_y$：

$$
\frac{\partial L}{\partial z_k}
=
-\frac{1}{\hat p_y}\frac{\partial \hat p_y}{\partial z_k}
=
-(\delta_{ky} - \hat p_k)
=
\hat p_k - \delta_{ky}
$$

写成向量就是 $\nabla_z L = \hat p - y_{\text{onehot}}$。**预测减目标**——形式干净，更重要的是它**不饱和**。

对比"sigmoid + MSE"的梯度：

$$
\frac{\partial L_{\mathrm{MSE}}}{\partial z}
\propto
(\sigma(z) - y)\,\sigma'(z)
$$

含一个 $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 因子——当 logit 很大或很小时 $\sigma'(z) \to 0$，梯度被压扁。这就是为什么早期神经网络分类用 MSE 收敛慢、改用 CE 后训得动的根源。

> CE 的"魔力"不在 loss 形态本身，而在它和 softmax / sigmoid 的组合让梯度变成一个不饱和的差分。

---

## 5) 三者对比一览

| Loss | 噪声 / 输出分布假设         | 估计的统计量 | 梯度形态                              | 凸性                       | 鲁棒性             | 对偶正则化         |
| ---- | --------------------------- | ------------ | ------------------------------------- | -------------------------- | ------------------ | ------------------ |
| MSE  | $\mathcal{N}(0,\sigma^2)$ | 条件均值     | $\propto (f-y)$，线性               | 严格凸（线性模型有闭式解） | 对离群点敏感       | L2（高斯先验）     |
| MAE  | $\mathrm{Laplace}(0,b)$   | 条件中位数   | $\mathrm{sign}(f-y)$，恒定模长      | 凸但不光滑                 | 鲁棒，但难精细收敛 | L1（拉普拉斯先验） |
| CE   | Bernoulli / Categorical     | 类后验概率   | $\hat p - y$，与 softmax 配合不饱和 | 关于$z$ 凸               | —                 | —                 |

最后一列预告下一节：噪声分布决定 loss，**权重的先验分布**则决定正则化。

---

## 6) Bayesian / Maximum A Posteriori (MAP) 视角：从先验到正则化

把 MLE 推广一步：在似然之上再加一个权重先验 $p(\theta)$，做 MAP 估计

$$
\hat\theta_{\mathrm{MAP}}
=
\arg\max_\theta \bigl[\log p(D\mid\theta) + \log p(\theta)\bigr]
=
\arg\min_\theta \bigl[\mathcal{L}_{\mathrm{NLL}}(\theta) - \log p(\theta)\bigr]
$$

后面这一项 $-\log p(\theta)$ 就是**正则项**。两个最常见的先验：

* **高斯先验** $\theta \sim \mathcal{N}(0,\tau^2 I)$：

  $$
  -\log p(\theta) = \frac{1}{2\tau^2}\|\theta\|_2^2 + \text{const}
  \quad\Rightarrow\quad
  \text{L2 正则 / weight decay}
  $$
* **拉普拉斯先验** $\theta \sim \mathrm{Laplace}(0, b)$：

  $$
  -\log p(\theta) = \frac{1}{b}\|\theta\|_1 + \text{const}
  \quad\Rightarrow\quad
  \text{L1 正则 / Lasso（产生稀疏解）}
  $$

> Weight decay 不是工程技巧，而是高斯先验的 NLL；L1 稀疏化也不是直觉拍脑袋，而是拉普拉斯先验的 NLL。

这和前面 MSE / MAE 的对应关系完全平行：

| 看                   | 高斯    | 拉普拉斯 |
| -------------------- | ------- | -------- |
| 加在**噪声**上 | MSE     | MAE      |
| 加在**权重**上 | L2 正则 | L1 正则  |

四个块拼起来，就是经典监督学习的概率视角。

---

## 7) 几个常见的 loss 变体

### 7.1 Label smoothing

把 CE 的目标 one-hot 改为

$$
\tilde y_k
=
(1-\epsilon)\,\mathbb{1}[y=k] + \frac{\epsilon}{K}
$$

等价于在目标分布上加均匀先验，缓解过拟合与 logit 爆炸。从 KL 角度看就是：让模型不要把 KL 优化到极端的 0，而是和一个"软目标"对齐——分类器的置信度被显式压低。

### 7.2 异方差 MSE / aleatoric uncertainty

如果假设 $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2(x))$ 且让模型同时输出 $\sigma_i$，NLL 是：

$$
\sum_i \Bigl[\frac{(y_i - f_i)^2}{2\sigma_i^2} + \log\sigma_i\Bigr]
$$

第一项是加权 MSE：模型对自己更不确定的样本，自动允许更大残差；第二项 $\log\sigma_i$ 防止它把 $\sigma$ 调到无穷大作弊。这就是 Kendall & Gal 一系列工作里学到的 aleatoric uncertainty。

### 7.3 Focal loss：CE 在类别不平衡下的改造

在 detection / 长尾分类中，"易样本"已经被分对、$\hat p_y$ 接近 1，但因为数量太多，它们的 CE 梯度仍然主导更新，淹没难样本的信号。Focal loss 的修法：

$$
\mathcal{L}_{\mathrm{focal}}
=
-\alpha (1 - \hat p_y)^\gamma \log \hat p_y
$$

* $\hat p_y \to 1$ 时 $(1-\hat p_y)^\gamma \to 0$，**易样本的 loss 被快速压低**；
* $\hat p_y$ 还小时 $(1-\hat p_y)^\gamma$ 接近 1，难样本的梯度几乎不变；
* $\alpha$ 是类别加权，处理类别不平衡。

它**不是从某个新的 NLL 推出来的**，而是 CE 的工程改造——这点和 MSE/MAE/CE 的"分布假设 → loss"模式不同，值得明确区分。

### 7.4 Hinge loss / Support Vector Machine (SVM) 视角：另一条非概率路径

二分类 $y \in \{-1, +1\}$，模型输出实数 $z = f(x)$：

$$
\mathcal{L}_{\mathrm{hinge}}
=
\max(0,\ 1 - y z)
$$

* $yz \ge 1$：分类正确且 margin 足够，loss 为 0，**不再贡献梯度**——和 CE 永远有梯度形成鲜明对比。
* $yz < 1$：loss 与 margin violation 线性增长。

它走的是"几何 margin"路线，而不是 NLL 路线。CE 永远在乎"再让正确类概率更高一点"，hinge 在 margin 达标后就完全不管，因此它的解依赖于"支持向量"。

### 7.5 InfoNCE：本质是 K 类 CE，也是互信息的变分下界

对比学习中常见的 InfoNCE loss：

$$
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\log \frac{\exp(\mathrm{sim}(q, k^+)/\tau)}{\sum_{i=0}^{K} \exp(\mathrm{sim}(q, k_i)/\tau)}
$$

把 $K$ 个候选（1 个正样本 + $K-1$ 个负样本）当成 $K$ 个类别，正样本是"正确类"，分子分母就是 softmax，外面取 $-\log$ 就是 CE。**这是 InfoNCE 的浅层视角**。

**深层视角：互信息的变分下界**。把 query $q$ 看作随机变量 $X$、正样本 $k^+$ 看作 $Y$、$K-1$ 个负样本独立采自边缘分布 $p(Y)$。van den Oord et al. (CPC, 2018) 证明：

$$
\boxed{I(X; Y) \ge \log K - \mathcal{L}_{\mathrm{InfoNCE}}}
$$

其中 $I(X; Y)$ 是 query 与 positive 之间的**互信息 (Mutual Information, MI)**：

$$
I(X; Y)
=
\mathbb{E}_{p(x,y)}\!\Bigl[\log \frac{p(x, y)}{p(x)p(y)}\Bigr]
=
D_{\mathrm{KL}}\bigl(p(x,y)\,\|\,p(x)p(y)\bigr)
$$

**最小化 InfoNCE = 最大化 $I(X; Y)$ 的一个下界**。$K$ 越大下界越紧，这就是对比学习里"负样本越多越好"的根本原因——不只是让分类任务更难，而是直接抬高 MI 估计的下界紧度。

直觉：模型必须学到一个表征空间，让正样本对的相似度显著高于负样本。这种"可区分度"恰好就是 $X, Y$ 之间统计依赖的强度，即互信息。

> InfoNCE 一个名字两个身份：**浅看是 K 类 CE，深看是 $I(\text{query}; \text{positive})$ 的变分下界**——这也是它名字中"**Info**"的来源。这条线索会在 7 号 blog [《信息瓶颈》](./7-Information-Bottleneck.md) 里被系统化：从信息论角度看，MAE / MSE / KL / InfoNCE 全都来自一个统一目标——最大化保留信息、最小化冗余。

### 7.6 KL 作为 loss：知识蒸馏

直接把 forward KL 当作 loss：

$$
\mathcal{L}_{\mathrm{KD}}
=
D_{\mathrm{KL}}\!\bigl(\pi_T(\cdot \mid x)\ \|\ \pi_\theta(\cdot \mid x)\bigr)
$$

把 teacher 软分布当目标分布，learner 拟合它。这正好和 5 号 blog 里"on-policy distillation 中 token-level Forward KL + 上下文来自 student"那段对应——KL 在那里既是 loss 也是诊断 mode-covering / mode-seeking 倾向的工具。

---

## 8) 一些容易混淆的点

**为什么回归不直接用 CE / 分类不直接用 MSE？** 本质是分布假设错配：

* 用 CE 做回归 = 假设输出是离散的、$y$ 必须落在固定的几个 bin；如果你愿意把回归目标离散化（如 distributional Reinforcement Learning (RL)、DALL·E 把图像 token 化），CE 反而是有效的设计——前提是离散化。
* 用 MSE 做分类 = 假设标签噪声是高斯的；不仅分布假设错（标签是离散的），更要命的是和 sigmoid / softmax 配合时梯度饱和（4.4 节）。

**软标签的 CE 还是 NLL 吗？** 是。当目标 $p_i$ 不是 one-hot 而是软分布时，CE 已经不再等同于"-log 真实类概率"，但它仍然等于 $H(p_i, \hat p_i)$，以及（差一个常数）$D_{\mathrm{KL}}(p_i \| \hat p_i)$。换句话说，CE 在软标签下变成了"用模型分布去逼近软目标分布"的 forward KL，蒸馏和 label smoothing 都用到这一性质。

---

## 9) 一句话总结

MSE / MAE / CE **都是 NLL 的特例**——分别假设高斯、拉普拉斯、Categorical 噪声 / 输出分布。Loss 形式由分布假设决定，梯度性质由 loss 与最后一层激活的耦合决定，正则项由权重先验决定。**这三件事——noise prior、activation pairing、weight prior——是设计与诊断 loss 的三根主线**。

> 顺这条主线再往前走一步，就接到 5 号 blog 的 KL 视角：CE 就是 forward KL 的经验形式；KL 散度的方向选择，决定了模型是 mode-covering 还是 mode-seeking。
