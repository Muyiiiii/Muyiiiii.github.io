---
title: "Thought of Position Embedding"
date: 2026-02-04
---
Sinusoidal PE vs. RoPE

# Sinusoidal PE vs. RoPE：同样出现 `cos(ω(m−n))`，但一个“加法偏移”，一个“乘法调制”

很多资料里会看到一句话：

> **Sinusoidal PE**：通过点积产生 `cos(ω(m−n))` 的**加法偏移**。
> **RoPE**：通过旋转产生 `cos(ω(m−n))` 的**乘法调制**。

这句话的意思其实很简单：
两者都能把“相对距离 `(m−n)`”编码进 attention 打分，但**进入打分的方式不同**：

- Sinusoidal PE 更像是在 logits 上**额外加一项**（像 bias / offset）
- RoPE 更像是用距离对内容相似度做**门控/相位调制**（像 modulation）

下面用最小推导讲清楚。

## 1. Attention 打分回忆：我们到底在比较什么？

标准自注意力的打分（忽略缩放）：

$$
\text{score}(m,n) = q_m^\top k_n
$$

位置编码的目的：让模型能感知 **token m 与 token n 的相对位置差** `Δ = m−n`。

## 2. Sinusoidal PE：为什么叫“加法偏移”？

### 2.1 做法：把位置向量加到输入上（additive）

经典 Sinusoidal PE 做的是：

$$
h_m = x_m + p_m,\quad h_n = x_n + p_n
$$

然后再投影成 Q/K：

$$
q_m=W_Q h_m,\quad k_n=W_K h_n
$$

打分：

$$
\text{score}(m,n) = (W_Q(x_m+p_m))^\top (W_K(x_n+p_n))
$$

### 2.2 展开：你会得到“一堆相加的项”

展开后是四项相加（内容项 + 交叉项 + 位置项）：

$$
\text{score}(m,n)
= (W_Q x_m)^\top(W_K x_n)
+ (W_Q x_m)^\top(W_K p_n)
+ (W_Q p_m)^\top(W_K x_n)
+ (W_Q p_m)^\top(W_K p_n)
$$

其中最后一项是纯位置贡献。对 sin/cos 的经典配对维度，有关键恒等式：

$$
[\sin(m\omega),\cos(m\omega)]\cdot[\sin(n\omega),\cos(n\omega)]
= \cos(\omega(m-n))
$$

也就是说，某些情况下你可以把打分理解成：

$$
\text{score}(m,n) = \underbrace{\text{content}(m,n)}_{\text{内容相似度}} + \underbrace{\text{(一堆交叉项)}}_{\text{内容} \times \text{位置}} + \underbrace{\cos(\omega(m-n))}_{\text{距离相关的加法项}}
$$

### 2.3 “加法偏移”是什么？

直觉上：

- `content(m,n)` 是“内容相似度底座”
- `cos(ω(m−n))` 像一个**额外加上去的距离打分项**
- 它对 logits 起到“加分/扣分”的效果，类似 **bias/offset**

所以说 Sinusoidal PE 通过点积产生 `cos(ω(m−n))` 的**加法偏移**。

## 3. RoPE：为什么叫“乘法调制”？

### 3.1 做法：不加到 embedding 上，而是旋转 q/k（rotary）

RoPE 的思路是：对每个位置 `pos`，用一个旋转矩阵 `R(pos)` 作用在 `q,k` 上：

$$
q'_m = R(m)\,q_m,\quad k'_n = R(n)\,k_n
$$

打分使用旋转后的：

$$
\text{score}(m,n) = q_m'^\top k_n'
= (R(m)q_m)^\top (R(n)k_n)
$$

旋转矩阵满足：

$$
R(m)^\top R(n) = R(n-m)
$$

因此：

$$
q_m'^\top k_n' = q_m^\top R(n-m)\,k_n
$$

也就是说，打分只依赖相对距离 `Δ = n−m`。

### 3.2 关键：`cos(ωΔ)` 出现在“乘内容”的位置上

只看一个 2D 子空间（RoPE 就是把维度两两配对做旋转），有：

$$
R(\Delta)=
\begin{bmatrix}
\cos(\omega\Delta) & -\sin(\omega\Delta)\\
\sin(\omega\Delta) & \cos(\omega\Delta)
\end{bmatrix}
$$

于是：

$$
q^\top R(\Delta)k
= \cos(\omega\Delta)\,(q^\top k)
\;+\;
\sin(\omega\Delta)\,(q^\top J k)
$$

其中 \(J\) 是 90° 旋转算子（不重要，知道它也是内容相关项就行）。

注意看第一项：

- `cos(ωΔ)` **乘在** `q^T k` 上

这就不是“额外加一项”，而是“把内容相似度按距离做缩放/门控”，并且还混入一个 `sin(ωΔ)` 调制的内容项。

### 3.3 “乘法调制”是什么？

直觉上：

- 先有内容相似度 `q^T k`
- 距离项 `cos(ωΔ)` 像一个**随距离变化的滤波器/门控**
- 它会放大/抑制/改变内容相似度在打分中的贡献

所以说 RoPE 是通过旋转产生 `cos(ω(m−n))` 的**乘法调制**。

## 4. 一句话总结（把那句对比翻译成人话）

- **Sinusoidal PE**：距离信息更像“额外加到分数上”的一项（**additive offset**）。
- **RoPE**：距离信息更像“乘在内容相似度上、与内容耦合”的门控/相位调制（**multiplicative modulation**）。

## 5. 一个小直觉：为什么这区别重要？

- **加法偏移**（Sinusoidal）：距离项可以相对“独立”地影响 logits（像 bias）。
- **乘法调制**（RoPE）：距离项会与内容相似度绑定，形成更强的“内容×位置”耦合，更自然地表达相对位置关系。

（具体哪种更好取决于模型与任务）
