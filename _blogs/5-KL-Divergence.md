---
title: "KL Divergence：从熵到 Forward/Reverse KL"
date: 2026-04-25
---
Introduction of KL Divergence, entropy, cross entropy, and the difference between Forward KL and Reverse KL.


KL divergence（Kullback-Leibler divergence）衡量的是：**用一个分布 $Q$ 去近似另一个分布 $P$ 时，多付出了多少信息代价**。

通常在机器学习里，我们可以先这样记：

$$
P = \text{真实数据分布}
$$

$$
Q = \text{模型分布}
$$

也就是说，真实世界的数据来自 $P$，但模型只能给出一个近似分布 $Q$。

---

## 1) KL Divergence 的期望写法

KL divergence 最常用的写法是：

$$
D_{\mathrm{KL}}(P \| Q)
=
\mathbb{E}_{x \sim P}
\left[
\log \frac{P(x)}{Q(x)}
\right]
$$

如果写成离散形式，就是：

$$
D_{\mathrm{KL}}(P \| Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

这个式子的关键点是：**期望是对 $P$ 取的**。

也就是：

> 从真实分布 $P$ 里采样 $x$，然后计算模型分布 $Q$ 对这个样本解释得好不好。

把它拆开看：

$$
D_{\mathrm{KL}}(P \| Q)
=
\mathbb{E}_{x \sim P}[\log P(x)]
-
\mathbb{E}_{x \sim P}[\log Q(x)]
$$

其中 $\log P(x)$ 是真实分布自己的 log probability，$\log Q(x)$ 是模型对真实样本给出的 log probability。

---

## 2) 先理解熵：分布本身有多不确定

熵（entropy）可以理解成：**一个随机变量本身的不确定性有多大**。

离散分布 $P$ 的熵是：

$$
H(P)
=
-\sum_x P(x)\log P(x)
$$

也可以写成期望形式：

$$
H(P)
=
\mathbb{E}_{x \sim P}[-\log P(x)]
=
-
\mathbb{E}_{x \sim P}[\log P(x)]
$$

直观上，$-\log P(x)$ 是事件 $x$  发生时带来的信息量：

* 如果 $P(x)$ 很大，说明事件很常见，信息量小。
* 如果 $P(x)$ 很小，说明事件很罕见，信息量大。
* 熵就是平均意义下，一个样本会带来多少信息量。

几个简单例子：

* $P=[1,0]$ 时，结果完全确定，$H(P)=0$。
* $P=[0.5,0.5]$ 时，两个结果一样可能，不确定性最大。如果用 $\log_2$，熵是 1 bit。
* $P=[0.9,0.1]$ 时，结果比较容易猜，所以熵小于公平硬币。

---

## 3) KL、交叉熵和熵的关系

交叉熵（cross entropy）是：

$$
H(P,Q)
=
\mathbb{E}_{x \sim P}[-\log Q(x)]
$$

它表示：**如果真实样本来自 $P$，但我们用 $Q$ 去编码它，平均代价是多少**。

KL divergence 可以写成：

$$
D_{\mathrm{KL}}(P \| Q)
=
H(P,Q) - H(P)
$$

这个式子非常重要：

* $H(P)$ 是真实分布自己的最优编码代价。
* $H(P,Q)$ 是用模型分布 $Q$ 编码真实样本时的代价。
* 两者的差值就是：用错误分布 $Q$ 代替真实分布 $P$ 时，多出来的信息代价。

所以 KL divergence 永远非负：

$$
D_{\mathrm{KL}}(P \| Q) \ge 0
$$

只有当 $P=Q$ 时，这个额外代价才是 0。

---

## 4) KL 有方向：$D_{\mathrm{KL}}(P \| Q)$ 不等于 $D_{\mathrm{KL}}(Q \| P)$

KL divergence 最容易忽略的一点是：**它不是对称的**。

$$
D_{\mathrm{KL}}(P \| Q)
\ne
D_{\mathrm{KL}}(Q \| P)
$$

这两个方向看起来只是把 $P,Q$ 换了位置，但优化行为非常不一样。

---

## 5) Forward KL：$D_{\mathrm{KL}}(P \| Q)$

Forward KL 是：

$$
D_{\mathrm{KL}}(P \| Q)
=
\mathbb{E}_{x \sim P}
\left[
\log\frac{P(x)}{Q(x)}
\right]
$$

它的期望来自 $P$，所以它问的是：

> 真实样本来了，模型有没有给它足够高的概率？

如果某个地方真实概率 $P(x)$ 很大，但模型概率 $Q(x)$ 很小，那么：

$$
\frac{P(x)}{Q(x)}
$$

会很大，KL 惩罚也会很大。

所以 Forward KL 最讨厌的是：

> 真实数据里经常出现的区域，模型却不给概率。

也就是：

$$
P(x) \text{ 高},\quad Q(x) \text{ 低}
$$

Forward KL 会强迫模型覆盖真实分布的高概率区域，因此它通常被称为 **mode-covering**。

---

## 6) 为什么 Forward KL 是 mode-covering

假设真实分布 $P$ 有两个峰，也就是两个 mode：

$$
P = \text{两个高概率区域}
$$

如果模型 $Q$ 只覆盖左边的 mode，却漏掉右边的 mode，那么在右边那个区域：

$$
P(x) \text{ 很大},\quad Q(x) \text{ 很小}
$$

于是：

$$
\log\frac{P(x)}{Q(x)}
$$

会变得很大。

所以为了降低 $D_{\mathrm{KL}}(P \| Q)$，模型必须尽量覆盖所有真实数据可能出现的区域。

如果 $Q$ 的表达能力有限，比如只能用一个 Gaussian 拟合双峰分布，那么 Forward KL 往往会让这个 Gaussian 放在两个峰中间，并且方差变大。这样做虽然可能会把概率放到两个峰中间的低密度区域，但它至少避免漏掉任何一个真实 mode。

一句话：

> Forward KL 宁愿覆盖得宽一点，也不要漏掉真实数据的 mode。

---

## 7) Reverse KL：$D_{\mathrm{KL}}(Q \| P)$

Reverse KL 是：

$$
D_{\mathrm{KL}}(Q \| P)
=
\mathbb{E}_{x \sim Q}
\left[
\log\frac{Q(x)}{P(x)}
\right]
$$

这次期望来自 $Q$，所以它问的是：

> 模型生成的样本，在真实分布下靠不靠谱？

如果模型 $Q$ 在某个地方放了很大概率，但真实分布 $P$ 在那里概率很低，那么：

$$
Q(x) \text{ 高},\quad P(x) \text{ 低}
$$

Reverse KL 会给出很大惩罚。

所以 Reverse KL 最讨厌的是：

> 模型把概率放到真实数据很少出现的地方。

因此它倾向于让模型待在最安全、最确定的高概率区域里，而不是冒险覆盖所有区域。这就是 **mode-seeking**。

---

## 8) 为什么 Reverse KL 是 mode-seeking

还是假设真实分布 $P$ 有两个 mode，而模型 $Q$ 只能用一个 Gaussian 去拟合。

如果 $Q$ 试图同时覆盖两个 mode，它很可能需要把概率也放到两个 mode 中间。但两个 mode 中间可能是 $P(x)$ 很低的区域。

这对 Reverse KL 很危险，因为在那里会出现：

$$
Q(x) \text{ 不小},\quad P(x) \text{ 很小}
$$

于是：

$$
\log\frac{Q(x)}{P(x)}
$$

会很大。

所以 Reverse KL 的最优策略经常是：只选择其中一个 mode，集中拟合它，避免把概率放到真实低密度区域。

一句话：

> Reverse KL 宁愿少覆盖一些真实 mode，也不要生成到真实低概率区域。

---

## 9) 极端情况：为什么两个方向差别这么大

对于 Forward KL：

$$
P(x)>0,\quad Q(x)=0
$$

会导致：

$$
\log\frac{P(x)}{0}
$$

趋向无穷大。

所以 Forward KL 会说：

> 真实分布有概率的地方，模型绝对不能给 0。

这就是它强迫 $Q$ 覆盖 $P$ 的原因。

反过来，对于 Reverse KL：

$$
Q(x)>0,\quad P(x)=0
$$

会导致：

$$
\log\frac{Q(x)}{0}
$$

趋向无穷大。

所以 Reverse KL 会说：

> 真实分布不可能出现的地方，模型绝对不能生成。

这就是它更保守、更 mode-seeking 的原因。

---

## 10) 用一个生活化例子理解

假设真实分布 $P$ 是“学生喜欢吃什么”：

* 50% 喜欢中餐。
* 50% 喜欢韩餐。
* 几乎没人喜欢某个奇怪口味。

现在模型 $Q$ 要推荐餐厅。

Forward KL 会想：

> 真实学生喜欢中餐和韩餐，所以我两个都要覆盖。

它宁愿推荐范围广一点，中餐、韩餐，甚至一些夹在中间的泛亚洲菜。它的目标是不要漏掉任何真实受欢迎的选择。

Reverse KL 会想：

> 我千万不要推荐学生不喜欢的东西。

如果它不确定韩餐是否安全，它可能只推荐中餐。它宁愿保守一点，只抓住一个非常确定的高概率区域。

---

## 11) 在机器学习中的理解

### 分类任务和交叉熵

分类任务里的 cross entropy 本质上接近：

$$
D_{\mathrm{KL}}(P_{\text{data}} \| P_\theta)
$$

也就是要求模型对真实标签给出高概率。

如果真实标签是类别 $y$，训练目标就是让模型增大：

$$
P_\theta(y \mid x)
$$

这和 Forward KL 的直觉一致：真实数据出现的地方，模型必须覆盖。

### Variational Inference 和 VAE

在 variational inference 里，我们经常有真实后验：

$$
p(z \mid x)
$$

但它通常很难直接计算，所以用一个简单分布：

$$
q_\phi(z \mid x)
$$

去近似它。

很多传统 VI 最小化的是：

$$
D_{\mathrm{KL}}(q_\phi(z \mid x) \| p(z \mid x))
$$

这是 Reverse KL。

因此如果真实后验 $p(z \mid x)$ 有多个 mode，近似后验 $q_\phi(z \mid x)$ 可能只选择其中一个 mode，而不是覆盖所有可能解释。这会让后验近似偏保守，并且可能低估不确定性。

### 生成模型和 mode collapse

如果优化行为太偏 Reverse KL，模型会倾向于只生成少数安全样本，而不是覆盖完整的数据多样性。

这和生成模型里的 mode collapse 直觉很接近：模型只抓住一部分 mode，生成结果看起来稳定，但 diversity 下降。

### On-policy distillation

在 LLM distillation 里，通常有两个模型：

$$
\pi_T = \text{teacher policy}
$$

$$
\pi_\theta = \text{student policy}
$$

distillation 的目标是让 student 模仿 teacher。但这里最容易混淆的一点是：**KL 的方向** 和 **样本从哪里来** 是两件事。

#### Offline / off-policy distillation

普通的离线蒸馏通常在固定数据集、人工答案、或者 teacher 生成的 prefix 上训练 student。设当前上下文是 $c$，token-level loss 常写成：

$$
\mathcal{L}_{\text{off}}
=
\mathbb{E}_{c \sim \mathcal{D}}
\left[
D_{\mathrm{KL}}
\left(
\pi_T(\cdot \mid c)
\|
\pi_\theta(\cdot \mid c)
\right)
\right]
$$

这里 teacher 是 $P$，student 是 $Q$，所以这是 token-level 的 Forward KL：

$$
D_{\mathrm{KL}}(\pi_T \| \pi_\theta)
$$

它会惩罚 student 漏掉 teacher 认为可能的 token，因此比较接近 mode-covering。

但问题是：上下文 $c$ 来自固定数据分布 $\mathcal{D}$，不一定是 student 自己推理时会走到的状态。也就是说，student 在训练时学的是“别人走到的 prefix”，但推理时面对的是“自己生成出来的 prefix”。这会产生 train-test mismatch，也就是常说的 exposure bias。

#### On-policy distillation 的关键变化

On-policy distillation 会先让 student 自己采样：

$$
y \sim \pi_\theta(\cdot \mid x)
$$

于是训练上下文变成：

$$
c_t = (x, y_{<t})
$$

也就是 student 自己生成出来的 prefix。然后再让 teacher 在这些 student-visited contexts 上给出 next-token 分布：

$$
\mathcal{L}_{\text{on}}
=
\mathbb{E}_{x,\ c_t \sim \pi_\theta}
\left[
D_{\mathrm{KL}}
\left(
\pi_T(\cdot \mid c_t)
\|
\pi_\theta(\cdot \mid c_t)
\right)
\right]
$$

注意这个式子有两层含义：

* 在每一个已经访问到的上下文 $c_t$ 上，token-level KL 仍然是 Forward KL：teacher 到 student。
* 但是这些上下文 $c_t$ 是由 student 自己采样出来的，所以外层期望是 on-policy 的，来自 $\pi_\theta$。

因此 on-policy distillation 的直觉是：

> student 自己走到哪里，就让 teacher 在哪里纠正它。

这比纯 offline distillation 更贴近推理时的状态分布。如果 student 生成了一个有点歪的 prefix，teacher 仍然可以在这个 prefix 上告诉它下一步应该怎么走，所以它更擅长修正 student 自己会犯的错误。

#### 为什么它又有 Reverse KL 的味道

如果从完整序列分布的角度看，student 先采样完整输出 $y$，然后我们比较 student 和 teacher 对这个输出的概率：

$$
\mathbb{E}_{y \sim \pi_\theta}
\left[
\log \pi_\theta(y \mid x)
-
\log \pi_T(y \mid x)
\right]
$$

这正是：

$$
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot \mid x)
\|
\pi_T(\cdot \mid x)
\right)
$$

也就是 Reverse KL：student 到 teacher。

所以 on-policy distillation 经常会有一种混合性质：

* 局部看，每个 visited context 上可能是在做 Forward KL distillation。
* 全局看，因为样本来自 student，它更关注 student 自己会生成的区域。
* 如果 student 从来不采到 teacher 的某个 mode，那么这个 mode 很可能不会被充分学习。

这就是为什么 on-policy distillation 往往更像 **让 student 在自己当前会走的区域里变得更像 teacher**，而不是保证 student 覆盖 teacher 的所有行为模式。

#### 实际影响

On-policy distillation 的优点是：

* 它减少 exposure bias，因为训练上下文来自 student 自己的推理分布。
* 它能让 teacher 针对 student 的真实错误状态提供监督。
* 它适合迭代式改进 student，因为每一轮都可以重新采样当前 student 的行为。

它的风险是：

* 如果 student 初始策略太窄，它可能永远采不到 teacher 的某些 mode。
* 如果只用 on-policy samples，训练会强化 student 当前分布附近的行为，diversity 可能下降。
* 如果 student 早期质量很差，teacher 会花很多监督信号在纠正低质量 prefix 上，而不是传授 teacher 的完整能力。

所以实际训练里经常会混合两类数据：

* 用 off-policy / teacher-generated 数据保证覆盖 teacher 的主要能力和多样性。
* 用 on-policy / student-generated 数据修正 student 自己推理时会遇到的问题。

一句话总结：

> Offline distillation 更像让 student 覆盖 teacher 已展示出来的行为；on-policy distillation 更像让 student 在自己会到达的状态上被 teacher 纠正。前者更 mode-covering，后者更贴近推理分布，但也更容易带有 mode-seeking 的偏向。

---

## 12) 最关键的记忆方式

Forward KL：

$$
D_{\mathrm{KL}}(P \| Q)
$$

期望来自 $P$，所以它站在真实样本视角：

> 真实样本来了，模型有没有给它概率？

因此：

$$
\text{Forward KL} = \text{mode-covering}
$$

Reverse KL：

$$
D_{\mathrm{KL}}(Q \| P)
$$

期望来自 $Q$，所以它站在模型样本视角：

> 模型生成的样本，在真实分布下靠谱吗？

因此：

$$
\text{Reverse KL} = \text{mode-seeking}
$$

最后一句话：

> Forward KL 怕漏掉真实 mode，所以倾向于覆盖所有可能性；Reverse KL 怕生成到低概率区域，所以倾向于选择一个最安全的 mode。
