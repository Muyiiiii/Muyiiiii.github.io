---
title: "Position Embedding"
date: 2026-02-03
---
About Position Embedding: Sinusoidal PE & RoPE

## 1. 为什么需要位置编码？

Transformer架构中的Self-Attention机制本质上是**置换不变的(permutation invariant)**。给定一个序列，无论token的顺序如何打乱，Attention的计算结果都是相同的（只是顺序不同）。

考虑输入序列 $X = [x_1, x_2, ..., x_n]$，Self-Attention的计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这个计算过程中，$x_i$ 和 $x_j$ 之间的关系只取决于它们的内容，而与位置无关。但在自然语言中，"我爱你"和"你爱我"的语义完全不同，因此我们需要**位置编码(Position Embedding)**来引入位置信息。

## 2. Sinusoidal Position Embedding (Sin/Cos位置编码)

### 2.1 基本思想

Transformer原论文提出使用正弦和余弦函数来编码位置：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：

- $pos$：token在序列中的位置（0, 1, 2, ...）
- $i$：维度索引（0, 1, 2, ..., $d_{model}/2 - 1$）
- $d_{model}$：模型的隐藏维度
- 每个维度对的频率为：$\omega_i = \frac{1}{10000^{2i/d_{model}}}$
- 完整PE向量: $PE_{pos} = [\sin(pos \cdot \omega_0), \cos(pos \cdot \omega_0), \sin(pos \cdot \omega_1), \cos(pos \cdot \omega_1), ...]$

将sin和cos交替排列，得到每个位置的完整PE向量（$d_{model}=8$）：

### 2.2 规律总结

1. **低维度（$i$小）**：频率高，随pos快速振荡，周期短（$T_0 = 2\pi \approx 6.28$）
2. **高维度（$i$大）**：频率低，随pos缓慢变化，周期长（$T_3 = 2\pi \times 1000 \approx 6283$）
3. **不同pos的区分度**：低维提供细粒度区分，高维提供粗粒度区分
4. **类似进制编码**：低维如"秒针"快速转动，高维如"时针"缓慢转动

### 2.3 为什么选择Sin/Cos？

#### 2.3.1 线性变换角度

**关键性质：相对位置可以通过线性变换表示**

对于任意固定的偏移量 $k$，存在一个线性变换矩阵 $M$，使得：

$$
PE_{pos+k} = M \cdot PE_{pos}
$$

证明：利用三角函数的和角公式：

$$
\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)
$$

$$
\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)
$$

写成矩阵形式：

$$
\begin{bmatrix} \sin(pos + k) \\ \cos(pos + k) \end{bmatrix} = \begin{bmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix} \begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}
$$

这意味着模型可以学习到相对位置关系！

#### 2.3.2 三角函数角度

假设我们有两个位置：位置 $m$（Query 的位置）和位置 $n$（Key 的位置）。它们在某一频率 $\omega$ 下的二维位置编码向量分别为：

$$
PE_m = \begin{bmatrix} \sin(\omega m) \\ \cos(\omega m) \end{bmatrix}, \quad PE_n = \begin{bmatrix} \sin(\omega n) \\ \cos(\omega n) \end{bmatrix}
$$

当 Transformer 计算 Attention 时，会进行 $Q$ 和 $K$ 的点积。如果我们只看位置编码部分的贡献，其计算如下：

$$
PE_m \cdot PE_n = \sin(\omega m)\sin(\omega n) + \cos(\omega m)\cos(\omega n)
$$

根据三角恒等式中的**余弦差角公式**：

$$
\cos(A - B) = \cos A \cos B + \sin A \sin B
$$

我们将 $A = \omega m$ 和 $B = \omega n$ 代入：

$$
PE_m \cdot PE_n = \cos(\omega m - \omega n)
$$

#### 2.3.3 结果分析：提取相对距离

令相对距离为 $k = m - n$，则点积结果变为：

$$
PE_m \cdot PE_n = \cos(\omega k)
$$

**这个推导告诉我们：**

1. **绝对位置消失了**：结果中不再含有 $m$ 或 $n$，只剩下它们的差值 $k$。
2. **距离编码**：点积的结果直接反映了两个位置之间的“角度差”。
   - 当 $k=0$（同一个位置）时，$\cos(0) = 1$，相关性最强。
   - 当 $k$ 增大时，$\cos(\omega k)$ 会随之变化。由于存在多个频率 $\omega$，这些 $\cos$ 值叠加在一起，形成了一个能唯一标识距离 $k$ 的“特征指纹”。

**$k$ 并不是一个序号，而是一个“频率通道”。** 它把位置这个抽象概念转化为了模型可以理解的、具有物理意义的多尺度信号。

**经典 PE：维度越靠后，频率越低（更慢、更长程）**

它这样排，是为了给模型一个从短程到长程的多尺度基底；顺序不是本质，但这样最方便、最稳定。本质上：**放哪一维其实没那么重要**，因为后面有线性层 $W_Q,W_K$ 会把维度混合；你把频率顺序打乱，模型理论上也能学。

### 2.4 Sin/Cos位置编码的局限性

1. **加法方式混合**：位置信息通过 $x + PE$ 加入，容易与语义信息混淆
2. **外推能力有限**：虽然理论上可以外推，但实际效果有限
3. **绝对位置编码**：本质上还是编码绝对位置

## 3. RoPE (Rotary Position Embedding)

### 3.1 核心思想

当我们计算位置 $m$ 的 $Q$ 和位置 $n$ 的 $K$ 的内积时：

$$
\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = \mathbf{q}^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}
$$

RoPE的核心思想是：**通过旋转矩阵将位置信息编码到Attention的Query和Key中，使得内积只依赖于相对位置**。

目标：设计一个位置编码函数 $f(x, pos)$，使得：

$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m-n)
$$

即Query和Key的内积只与它们的**相对位置** $m-n$ 有关。

### 3.2 数学推导

#### 二维情况

从最简单的二维情况开始。设 $q = [q_0, q_1]^T$，我们希望找到函数 $f$。

**旋转矩阵**：将向量 $q$ 旋转角度 $\theta$：

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

定义位置编码函数：

$$
f(q, m) = R(m\theta) \cdot q = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \end{bmatrix}
$$

展开计算：

$$
f(q, m) = \begin{bmatrix} q_0\cos(m\theta) - q_1\sin(m\theta) \\ q_0\sin(m\theta) + q_1\cos(m\theta) \end{bmatrix}
$$

#### 验证相对位置性质

计算 $f(q, m)$ 和 $f(k, n)$ 的内积：

$$
\langle f(q, m), f(k, n) \rangle = f(q, m)^T \cdot f(k, n)
$$

利用旋转矩阵的性质：

$$
R(\alpha)^T = R(-\alpha) = \begin{bmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{bmatrix}
$$

所以我们有：

$$
\langle f(q, m), f(k, n) \rangle = f(q, m)^T \cdot f(k, n)= q^T R(m\theta)^T R(n\theta) k = q^T R(-m\theta) R(n\theta) k = q^T R((n-m)\theta) k
$$

结果只依赖于 $(n-m)$，即**相对位置**！

### 3.3 高维扩展

对于 $d$ 维向量（$d$ 为偶数），将其分成 $d/2$ 个二维子空间，每个子空间独立旋转：

$$
R_{\Theta, m} = \begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

其中频率设定与Sinusoidal类似：

$$
\theta_i = 10000^{-2i/d}
$$

### 3.4 详细计算示例

假设 $d = 4$，位置 $m = 2$，输入向量 $q = [1, 2, 3, 4]^T$

**Step 1：计算各频率**

$$
\theta_0 = 10000^{-0/4} = 1
$$

$$
\theta_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01
$$

**Step 2：计算旋转角度**

$$
m\theta_0 = 2 \times 1 = 2
$$

$$
m\theta_1 = 2 \times 0.01 = 0.02
$$

**Step 3：构建旋转矩阵**

$$
R_{\Theta, 2} = \begin{bmatrix}
\cos(2) & -\sin(2) & 0 & 0 \\
\sin(2) & \cos(2) & 0 & 0 \\
0 & 0 & \cos(0.02) & -\sin(0.02) \\
0 & 0 & \sin(0.02) & \cos(0.02)
\end{bmatrix}
$$

数值代入（$\cos(2) \approx -0.416$，$\sin(2) \approx 0.909$，$\cos(0.02) \approx 1.0$，$\sin(0.02) \approx 0.02$）：

$$
R_{\Theta, 2} \approx \begin{bmatrix}
-0.416 & -0.909 & 0 & 0 \\
0.909 & -0.416 & 0 & 0 \\
0 & 0 & 1.0 & -0.02 \\
0 & 0 & 0.02 & 1.0
\end{bmatrix}
$$

**Step 4：应用旋转**

$$
f(q, 2) = R_{\Theta, 2} \cdot q = \begin{bmatrix}
-0.416 \times 1 + (-0.909) \times 2 \\
0.909 \times 1 + (-0.416) \times 2 \\
1.0 \times 3 + (-0.02) \times 4 \\
0.02 \times 3 + 1.0 \times 4
\end{bmatrix} = \begin{bmatrix}
-2.234 \\
0.077 \\
2.92 \\
4.06
\end{bmatrix}
$$

### 3.5 高效实现

直接构建完整旋转矩阵计算量大，实际实现中使用等价的逐元素运算：

$$
f(x, m) = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \end{bmatrix} \odot \begin{bmatrix} \cos(m\theta_0) \\ \cos(m\theta_0) \\ \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \end{bmatrix} + \begin{bmatrix} -x_1 \\ x_0 \\ -x_3 \\ x_2 \\ \vdots \end{bmatrix} \odot \begin{bmatrix} \sin(m\theta_0) \\ \sin(m\theta_0) \\ \sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \end{bmatrix}
$$

PyTorch伪代码：

```python
def apply_rope(x, cos, sin):
    # x: [batch, seq_len, num_heads, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]  # 分离奇偶维度
    # 旋转操作
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + x_rotated * sin
```

## 4. Sin/Cos vs RoPE 对比

| 特性           | Sinusoidal PE         | RoPE                 |
| -------------- | --------------------- | -------------------- |
| 编码方式       | 加法 (x + PE)         | 乘法（旋转）         |
| 位置信息注入点 | Embedding层           | Attention的Q, K      |
| 相对位置       | 隐式（需模型学习）    | 显式（内积直接反映） |
| 外推能力       | 较弱                  | 较强                 |
| 计算开销       | 低                    | 中等                 |
| 应用           | 原始Transformer, BERT | LLaMA, GPT-NeoX      |

---

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
