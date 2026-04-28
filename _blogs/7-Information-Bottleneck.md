---
title: "7\\. 信息瓶颈视角：从互信息推导 MSE / MAE / KL / InfoNCE"
date: 2026-04-28
---
Deriving MSE, MAE, KL, and InfoNCE losses from the Information Bottleneck framework, unified via mutual information.


6 号 [《损失函数推导》](/blogs/6-Loss-Functions/) 顺着"假设噪声 → 写出 NLL → 得到 loss"那条主线，把 Mean Squared Error (MSE) / Mean Absolute Error (MAE) / Cross Entropy (CE) 推了一遍。这篇换一个视角：**所有这些 loss 也都可以从信息论一根根线扯出来**——具体说，从**互信息 (Mutual Information, MI)** 和 **Information Bottleneck (IB)** 这两个工具出发，MSE / MAE / KL / InfoNCE 会在同一张框架图里各自找到自己的位置。

---

## 0) 重点速览

* **MI 在 loss 推导里有三种身份**：
  * **作为目标**：最大化 $I(Z;Y)$ 在不同噪声假设下退化为 MSE / MAE / CE（NLL 视角的等价物）；
  * **作为正则**：上界 $I(X;Z)$ 给出 KL 散度——这就是 Variational Information Bottleneck (VIB) 和 VAE 里 KL 项的来源；
  * **作为度量**：$I(X;Y)$ 自身的变分下界给出 **InfoNCE**——对比学习的根基。
* **IB 框架** $\beta I(X;Z) - I(Z;Y)$ 把"压缩"和"预测"塞进一个目标里——前者要 KL，后者要 NLL，正好对应监督学习里"正则 + 损失"两块。
* **InfoNCE 的"Info"两字不是装饰**：它是一个 MI 下界，下界紧度直接由负样本数 $K$ 决定。
* 把 6 号 blog 与本篇拼起来：6 号说"loss 来自分布假设"，本篇说"分布假设之下还藏着一个统一的 MI 目标"。

---

## 1) 信息论速览：熵 → 条件熵 → 互信息

**熵 (entropy)**：

$$
H(X) = -\mathbb{E}_{p(x)}[\log p(x)]
$$

衡量 $X$ 自身的不确定性（参见 5 号 [《KL Divergence》](/blogs/5-KL-Divergence/) 第 2 节）。

**条件熵 (conditional entropy)**：

$$
H(Y \mid X) = -\mathbb{E}_{p(x,y)}[\log p(y \mid x)]
$$

知道 $X$ 之后 $Y$ 还剩多少不确定性。

**互信息 (Mutual Information, MI)**：

$$
\boxed{
I(X; Y)
=
H(Y) - H(Y \mid X)
=
\mathbb{E}_{p(x,y)}\!\Bigl[\log \frac{p(x, y)}{p(x)\,p(y)}\Bigr]
=
D_{\mathrm{KL}}\bigl(p(x,y)\,\|\,p(x)\,p(y)\bigr)
}
$$

三种等价写法对应三种直觉：

* **不确定性减少**：知道 $X$ 让 $Y$ 的不确定性减少了多少（$H(Y) - H(Y|X)$）。
* **联合 vs 独立**：联合分布与"假设独立"的乘积分布之间的 KL——$X, Y$ 越独立 MI 越小。
* **预测可分离性**：高 MI 意味着用 $X$ 能很好预测 $Y$。

性质：

* $I(X;Y) \ge 0$，等号成立当且仅当 $X \perp Y$。
* $I(X;Y) = I(Y;X)$，对称。
* **Data Processing Inequality (DPI)**：若 $X \to Z \to Y$ 构成马尔可夫链，则 $I(X; Y) \ge I(Z; Y)$——经过任何处理都不会增加 MI。

> MI 把"$X$ 与 $Y$ 的统计依赖强度"压进一个数。它就是后面所有推导的中心物理量。

---

## 2) Information Bottleneck：把"压缩"和"预测"塞进一个目标

Tishby, Pereira & Bialek (1999) 提出：在监督学习里，从 $X$ 学一个表征 $Z$，希望

* $Z$ **压缩**掉 $X$ 中与 $Y$ 无关的细节——$I(X; Z)$ 要小；
* $Z$ **保留**预测 $Y$ 所需的信息——$I(Z; Y)$ 要大。

合起来就是 IB 目标（取负后做最小化）：

$$
\boxed{
\mathcal{L}_{\mathrm{IB}}
=
\beta\,I(X; Z) - I(Z; Y)
}
$$

* $\beta > 0$ 控制压缩与预测之间的权衡。
* 这是经典 **rate-distortion** 思想的直接延伸：$I(X;Z)$ 是 *rate*（用了多少信息），$I(Z;Y)$ 是 *negative distortion*（保住了多少与 $Y$ 相关的信号）。

把它套进监督学习的"输入 → 表征 → 输出"链：

```
 X  ──encoder──▶  Z  ──decoder──▶  Y
 │                                  │
 └──── 压缩  I(X;Z)（小）            │
                  └── 预测  I(Z;Y)（大）
```

**接下来三节会做的事情**：

* §3 / §4：把"最大化 $I(Z;Y)$"在不同条件分布假设下展开，分别得到 MSE 和 MAE。
* §5：把"最小化 $I(X;Z)$"用变分上界改写，得到 KL（VIB / VAE 的 KL 项）。
* §6：上面的目标都假设我们能算 MI；现实里不能，于是用变分下界估计——这就是 InfoNCE。

---

## 3) 从 MI 推导 MSE：高斯条件

要"最大化 $I(Z; Y)$"。利用恒等式 $I(Z;Y) = H(Y) - H(Y \mid Z)$，而 $H(Y)$ 是数据分布决定的常数（与参数无关），所以：

$$
\arg\max_\theta I(Z;Y)
=
\arg\min_\theta H(Y \mid Z)
=
\arg\min_\theta \bigl(-\mathbb{E}_{p(z, y)}[\log p_\theta(y \mid z)]\bigr)
$$

也就是**最大化 MI ≡ 最小化条件 NLL**。

现在给 $p_\theta(y \mid z)$ 选一个参数族。最常见的选择：高斯，均值由 $z$ 经过解码器 $g_\theta$ 给出，方差固定为 $\sigma^2$：

$$
p_\theta(y \mid z)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\Bigl(-\frac{(y - g_\theta(z))^2}{2\sigma^2}\Bigr)
$$

代回：

$$
-\log p_\theta(y \mid z)
=
\frac{(y - g_\theta(z))^2}{2\sigma^2}
+
\text{const}
$$

求和、丢掉常数、把 $\frac{1}{2\sigma^2}$ 吸进学习率，得到

$$
\boxed{
\arg\max_\theta I(Z; Y)
\stackrel{\text{Gaussian}}{=}
\arg\min_\theta \frac{1}{N}\sum_i (y_i - g_\theta(z_i))^2
=
\mathcal{L}_{\mathrm{MSE}}
}
$$

> **MSE = "最大化 MI" 在高斯条件下的特例**。这跟 6 号 blog 里"MSE = NLL 在高斯噪声下的特例"是同一件事的两种说法——MI 视角只是把"最优条件密度"显式拎出来了。

---

## 4) 从 MI 推导 MAE：拉普拉斯条件

把 §3 里的 $p_\theta(y \mid z)$ 换成拉普拉斯：

$$
p_\theta(y \mid z) = \frac{1}{2b}\exp\!\Bigl(-\frac{|y - g_\theta(z)|}{b}\Bigr)
$$

照搬同样的推导：

$$
-\log p_\theta(y \mid z)
=
\frac{|y - g_\theta(z)|}{b}
+
\text{const}
$$

得到

$$
\boxed{
\arg\max_\theta I(Z; Y)
\stackrel{\text{Laplace}}{=}
\arg\min_\theta \frac{1}{N}\sum_i |y_i - g_\theta(z_i)|
=
\mathcal{L}_{\mathrm{MAE}}
}
$$

> 把高斯换成 Categorical 就回到 CE。这三种 loss 不是各自孤立、而是**同一个 MI 目标 + 不同条件分布**。

把这三种情况拼起来：

| 假设的 $p(y \mid z)$ | $-\log p(y\mid z)$ 的形态 | MI 最大化退化为 |
|---|---|---|
| 高斯 $\mathcal{N}$ | $\propto (y - g(z))^2$ | MSE |
| 拉普拉斯 $\mathrm{Laplace}$ | $\propto |y - g(z)|$ | MAE |
| Categorical | $-\log \hat p_y$ | CE |

---

## 5) KL 从哪里来：IB 压缩项的变分上界

回到 IB 目标：$\beta I(X; Z) - I(Z; Y)$。$I(Z;Y)$ 已经在 §3 / §4 里变成 NLL；那 $I(X; Z)$ 呢？

直接算 $I(X; Z) = \mathbb{E}_{p(x, z)}\bigl[\log \frac{p(z \mid x)}{p(z)}\bigr]$ 需要边际 $p(z) = \mathbb{E}_{p(x)}[q(z \mid x)]$，对深度网络来说没办法闭式求出。**Variational Information Bottleneck (Alemi et al., 2017) 的解法**：换成一个可计算的上界——

$$
I(X; Z)
=
\mathbb{E}_{p(x)}\!\bigl[D_{\mathrm{KL}}(q(z \mid x)\,\|\,p(z))\bigr]
-
\underbrace{D_{\mathrm{KL}}(q(z) \,\|\, p(z))}_{\ge 0}
\le
\mathbb{E}_{p(x)}\!\bigl[D_{\mathrm{KL}}(q(z \mid x)\,\|\,p(z))\bigr]
$$

其中 $q(z \mid x)$ 是 encoder（学习的），$p(z)$ 是先验（如 $\mathcal{N}(0, I)$，固定的）。

代回 IB：

$$
\boxed{
\mathcal{L}_{\mathrm{VIB}}
=
\beta\,\mathbb{E}_{p(x)}\!\bigl[D_{\mathrm{KL}}(q(z \mid x)\,\|\,p(z))\bigr]
-
\mathbb{E}_{p(x, y)}\bigl[\log p_\theta(y \mid z)\bigr]
}
$$

读法：

* 后一项是**重建 / 预测损失**——按 §3 / §4，对 Gaussian / Laplace / Categorical 假设分别落到 MSE / MAE / CE。
* 前一项是 **encoder 与先验之间的 KL**——这就是**所有 VAE / VIB 类模型里那个 KL 项的来源**。

> **KL 不是被人为加进 loss 的正则项，而是 IB 压缩目标 $I(X;Z)$ 的可计算上界**。VAE 的 ELBO、$\beta$-VAE、VIB 全都是这套结构的不同实例。

把 §3–§5 拼起来，监督学习里"loss + 正则"的双层结构就有了一个统一的来源：

$$
\underbrace{\text{MSE / MAE / CE}}_{\text{来自 } -I(Z;Y) \text{ 用具体条件分布展开}}
\quad+\quad
\beta\cdot\underbrace{D_{\mathrm{KL}}(q(z\mid x)\,\|\,p(z))}_{\text{来自 } I(X;Z) \text{ 的变分上界}}
$$

---

## 6) InfoNCE：MI 的变分下界

§3–§5 默认 MI 算得出来。但实际中 $I(X; Y)$ 和 $I(Z; Y)$ 通常都是难算的——需要边际 $p(y)$ 或者 $p(z)$。表征学习需要一个**直接对 MI 求变分下界**的方法。这正是 InfoNCE（Oord et al., Contrastive Predictive Coding (CPC), 2018）做的事。

### 6.1 设定

考虑一批 $K$ 个候选 $\{y_1, \dots, y_K\}$，其中：

* 第 $1$ 个是来自联合分布 $p(x, y)$ 的"正样本" $y^+$；
* 其余 $K - 1$ 个是来自边际 $p(y)$ 的"负样本"。

定义一个评分函数 $f(x, y)$（一般是 query 与 key 嵌入的内积 / 相似度）。InfoNCE loss：

$$
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\,\mathbb{E}\!\left[
\log \frac{f(x, y^+)}{\sum_{i=1}^{K} f(x, y_i)}
\right]
$$

形式上就是一个 $K$ 类 softmax 分类——这是 6 号 blog 里那条"浅层"解释。

### 6.2 为什么它叫 "**Info**"NCE：MI 下界的证明

最优分类器的概率正比于"哪个候选真的来自联合分布"，可以写成

$$
\Pr(\text{positive index} = i \mid x, \{y_j\})
=
\frac{p(y_i \mid x) / p(y_i)}{\sum_{j} p(y_j \mid x)/p(y_j)}
$$

把式子代回 InfoNCE 的期望，再对 $K - 1$ 个负样本来源求期望，可以推出（细节略）：

$$
\boxed{
I(X; Y) \ge \log K - \mathcal{L}_{\mathrm{InfoNCE}}
}
$$

也就是：

$$
\mathcal{L}_{\mathrm{InfoNCE}}
\ge
\log K - I(X; Y)
$$

**最小化 InfoNCE = 最大化 $I(X; Y)$ 的一个下界**。直觉上：

* $f(x, y) \approx \frac{p(y \mid x)}{p(y)}$ 时分类器最优——这个比值正好是 MI 被积函数。
* 负样本数 $K$ 越大下界越紧；这就是 SimCLR / MoCo 里"队列越大效果越好"的直接依据。

### 6.3 一图看清三种 MI 下界 / 上界

| MI 操作 | 工具 | 用在哪里 |
|---|---|---|
| 上界 $I(X; Z)$ | $\mathbb{E}_{p(x)}[D_{\mathrm{KL}}(q(z\mid x)\|p(z))]$ | VIB / VAE 的 KL 正则 |
| 直接最大化 $I(Z; Y)$ | NLL（条件分布参数化） | MSE / MAE / CE |
| 下界 $I(X; Y)$ | InfoNCE / CPC | 对比学习 |

---

## 7) 一张图：四个 loss 在 MI / IB 下的位置

```
                    Information Bottleneck:
                    L_IB = β · I(X;Z)  −  I(Z;Y)
                            │                │
                            │                │
            上界（变分）     │                │  下式：固定条件密度
                            ▼                ▼
                  KL(q(z|x) ‖ p(z))      −E[log p(y|z)]
                  (VIB / VAE 的 KL 项)        │
                                              │   p(y|z) 取
                                              ▼
                              ┌───────────────┼────────────────┐
                              │               │                │
                          高斯              拉普拉斯       Categorical
                              │               │                │
                              ▼               ▼                ▼
                            MSE             MAE               CE


            另一条独立路径：MI 自己也需要被估计
                            │
                            ▼
                     I(X; Y) 的变分下界
                            │
                            ▼
                        InfoNCE  (= log K − L)
                        (CPC / 对比学习)
```

* **左侧**：IB 的"压缩"项 $I(X;Z)$ → 变分上界 → KL → VAE / VIB。
* **中间**：IB 的"预测"项 $-I(Z;Y)$ → 选条件密度族 → MSE / MAE / CE。
* **右侧**：当 MI 自己是估计目标时 → 变分下界 → InfoNCE。

---

## 8) 一句话总结

MSE / MAE / KL / InfoNCE **不是四条独立的 loss 路线，而是 MI 在三种角色下的不同落点**：

* **MI 当目标**：在 Gaussian / Laplace / Categorical 假设下变成 MSE / MAE / CE；
* **MI 当正则**：用变分上界变成 VAE / VIB 里的 KL 项；
* **MI 当度量**：用变分下界变成 InfoNCE。

> 把 6 号 blog 与本篇拼起来：6 号是"分布假设 → loss"的局部视角；本篇是"统一目标 (MI / IB) → 不同实现"的全局视角。两条路殊途同归：**所有这些 loss 背后都是同一件事——让模型尽可能保留信号、丢掉噪声**。
