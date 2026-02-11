---
title: "Group Relative Policy Optimization (GRPO)"
date: 2026-02-11
---
Introduction of Group Relative Policy Optimization (GRPO)


把 **GRPO** 讲清楚：**为什么 PPO 在 LLM 上不好用**、**GRPO 怎么去掉 Critic**、**Group Relative Advantage 怎么算**、以及 **GRPO 的 loss 长什么样、和 PPO 有什么区别**。默认已经读过上一篇 PPO，知道 ratio、clip、Advantage 这些概念。

---

## 1) 动机：PPO 用在 LLM 上有什么问题

回忆 PPO 的训练流程：我们需要 **四个模型**（或至少三个）：

| 模型 | 作用 |
| --- | --- |
| Policy（Actor） | 生成文本的 LLM，就是我们要训练的 |
| Reference model | 冻结的初始策略，用来算 KL 惩罚 |
| Reward model | 给生成文本打分 |
| Value model（Critic） | 估计 $V(s_t)$，用来算 Advantage |

问题出在 **Value model**：

* **显存翻倍**：Critic 和 Policy 通常同等规模（都是 LLM），相当于要同时维护两个大模型的参数和优化器状态
* **训练不稳定**：token-level 的 value estimation 在语言任务上很难学好——自然语言不像游戏有明确的中间状态价值，Critic 经常估不准，导致 Advantage 噪声很大
* **工程复杂**：需要同时调 Actor 和 Critic 的学习率、loss 权重、训练节奏等，超参数空间翻倍

**GRPO 的核心想法**：既然 Critic 这么难搞，能不能 **直接不要 Critic**，用一种更简单的方式估计 Advantage？

答案是：**对同一个问题采样一组（Group）回答，用组内奖励的相对排名来当 Advantage**。

---

## 2) GRPO 的核心：Group Relative Advantage

### 2.1 采样一组输出

对每个输入 prompt $q$，用当前策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个输出：

$$
\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)
$$

然后用 Reward Model 给每个输出打分，得到 $\{r_1, r_2, \ldots, r_G\}$。

### 2.2 组内标准化 → Advantage

**关键公式**（极其简单）：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}
$$

就是对这一组奖励做 **z-score 标准化**。

**直觉**：

* 不需要学一个 value function 来当 baseline——**组内均值天然就是 baseline**
* $\hat{A}_i > 0$：这个回答比同组平均好，应该被鼓励
* $\hat{A}_i < 0$：这个回答比同组平均差，应该被抑制
* 除以 std 保证不同 prompt 之间 Advantage 的尺度一致

### 2.3 为什么这样做合理？

回忆 PPO 里 Advantage 的本质：

$$
A(s,a) = Q(s,a) - V(s)
$$

$V(s)$ 是 baseline，减去它是为了降方差。GRPO 做的事：

* $Q(s,a)$ → 用 reward model 的打分 $r_i$ 近似（整个序列级别的奖励）
* $V(s)$ → 用同组采样的平均奖励 $\text{mean}(r)$ 近似

本质上是 **Monte Carlo baseline**：对同一个 prompt 采多条，用均值当 baseline。这在 REINFORCE 的理论里是完全合法的 variance reduction 技巧。

**和 PPO 的对比**：

| | PPO | GRPO |
| --- | --- | --- |
| Baseline | 学一个 $V_\phi(s_t)$ | 同组均值 $\text{mean}(r)$ |
| 粒度 | token-level | sequence-level |
| 额外模型 | 需要 Critic | 不需要 |
| Advantage 质量 | 取决于 Critic 好坏 | 取决于采样数 $G$ |

---

## 3) GRPO 的 Loss：和 PPO 长得很像，但有关键区别

### 3.1 Policy Loss（依然是 clip surrogate）

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\Big(
\rho_{i,t}(\theta)\,\hat{A}_i,\;
\text{clip}\big(\rho_{i,t}(\theta),\,1-\epsilon,\,1+\epsilon\big)\,\hat{A}_i
\Big)
$$

其中 ratio：

$$
\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

**注意关键区别**：

1. **$\hat{A}_i$ 是 sequence-level 的**：整个输出 $o_i$ 共用一个 Advantage（不像 PPO 里每个 token 有自己的 $A_t$）
2. **ratio 是 token-level 的**：每个 token 位置都有自己的 $\rho_{i,t}$，clip 也是逐 token 做的
3. **除以 $|o_i|$**：对序列长度做归一化，防止长回答 dominate loss

### 3.2 KL 惩罚（替代 Entropy bonus）

GRPO 不加 entropy bonus，而是直接加 **KL 散度惩罚**（相对于 reference model $\pi_{\text{ref}}$）：

$$
\mathcal{L}_{\text{KL}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} D_{\text{KL}}\big[\pi_\theta(\cdot \mid q, o_{i,<t}) \;\|\; \pi_{\text{ref}}(\cdot \mid q, o_{i,<t})\big]
$$

实际实现中，DeepSeek 用的是逐 token 的 **近似 KL**（不用算完整分布）：

$$
D_{\text{KL}}^{\text{approx}} = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1
$$

这个近似来自 $D_{\text{KL}}(P\|Q)$ 的一种单样本估计。$\frac{p}{q} - \log\frac{p}{q} - 1 \geq 0$，当且仅当 $p=q$ 时等于 0，所以它是合法的散度度量。

### 3.3 总 Loss

$$
\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \beta \, \mathcal{L}_{\text{KL}}
$$

**对比 PPO 的总 loss**：

| 项 | PPO | GRPO |
| --- | --- | --- |
| Policy loss | clip surrogate（token-level A） | clip surrogate（sequence-level A） |
| Value loss | $c_v(V_\phi - \hat{R})^2$ | **没有**（不需要 Critic） |
| 正则项 | entropy bonus $-c_e \mathcal{H}$ | KL penalty $\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ |

**为什么 GRPO 没有 entropy bonus？**

回忆 PPO 里加 entropy bonus 的目的：防止策略过早坍缩成确定性策略，保持探索。但在 GRPO 里，这件事已经被 **KL penalty** 覆盖了——KL 惩罚把策略锚定在 reference model 附近，而 reference model（通常是 SFT 后的模型）本身就有足够的输出多样性。只要我们不跑太远，策略自然不会坍缩。换句话说，KL penalty 同时起到了"防坍缩"和"防 reward hacking"两个作用，entropy bonus 就没必要再加了。而且 entropy 在 LLM 的巨大词表上算起来开销也不小，去掉反而更干净。

---

## 4) Outcome-level vs. Process-level Reward

上面讲的是最基本的 GRPO，用 **outcome-level reward**（对整个输出打一个分数）。但 GRPO 的框架也能支持 **process-level reward**（对推理过程的每一步打分）：

### 4.1 Outcome Supervision（Outcome ORM）

$$
\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
$$

整个序列共用一个 $\hat{A}_i$。

### 4.2 Process Supervision（Process PRM）

如果有 process reward model（对每个推理步骤 $k$ 打分 $r_i^{(k)}$），可以做更细粒度的 Advantage：

$$
\hat{A}_i^{(k)} = \frac{r_i^{(k)} - \text{mean}(r^{(k)})}{\text{std}(r^{(k)})}
$$

此时每一步有自己的 Advantage，属于该步的 token 共用这个值。这更接近 PPO 的 token-level credit assignment，但仍然不需要 Critic。

---

## 5) GRPO 的训练流程

1. **采样**：对一批 prompts $\{q_j\}$，每个用 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个输出
2. **打分**：用 Reward Model 对每个输出打分 $r_i$
3. **算 Advantage**：组内 z-score 标准化
4. **多 epoch 更新**：在同一批数据上做若干轮 SGD
   * 计算 token-level ratio $\rho_{i,t}$
   * clip surrogate loss
   * KL penalty
   * 合成总 loss，反向传播
5. **同步**：$\theta_{\text{old}} \leftarrow \theta$，进入下一轮采样

和 PPO 流程对比，**少了两步**：不需要算 $V(s_t)$，不需要跑 GAE。

---

## 6) GRPO 和 PPO 的完整对比

| 维度 | PPO（用于 LLM） | GRPO |
| --- | --- | --- |
| 需要的模型 | Policy + Critic + Reference + Reward | Policy + Reference + Reward |
| 显存开销 | 很大（Critic ≈ Policy 大小） | 约省一半（无 Critic） |
| Advantage 估计 | GAE（token-level，依赖 Critic） | 组内 z-score（sequence-level，无 Critic） |
| Credit assignment | token-level（精细但需 Critic 准确） | sequence-level（粗糙但稳定） |
| 训练稳定性 | 依赖 Critic 质量，调参多 | 更简单，但依赖采样数 $G$ |
| 适用场景 | 通用 RL / game | LLM alignment / reasoning |

---

## 7) 实践细节和"坑点"

* **采样数 $G$ 很关键**：$G$ 太小，均值 baseline 噪声大；$G$ 太大，计算开销大。DeepSeek-R1 里 $G=64$，DeepSeek-Math 里通常 $G \in [16, 64]$
* **当组内所有回答得分一样时**：$\text{std}=0$，除以 std 会爆。实现中需要加 $\epsilon$：$\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon}$，或者直接跳过这个 group
* **Reward hacking**：因为只看最终奖励，模型可能学到讨好 Reward Model 的"表面技巧"。KL penalty 和 reward model 质量是关键防线
* **长度偏差**：reward model 倾向给长回答高分 → 模型越写越长。归一化 loss 除以 $|o_i|$ 能缓解，但不能完全解决
* **KL 系数 $\beta$**：太大训不动（被锁死在 reference 附近），太小容易 reward hacking。常见做法是动态调整或者 warmup
* **温度采样**：采样 $G$ 个输出时的 temperature 会影响组内多样性。太低 → 回答太相似，Advantage 区分度小；太高 → 质量太差
* **和 PPO 的互补**：一些工作（如 DeepSeek-R1）先用 GRPO 大规模训练，再用 PPO 做精调。GRPO 省资源适合大规模训，PPO 精细调控适合最后阶段打磨

---

## 8) 从 REINFORCE 视角理解 GRPO

如果熟悉 REINFORCE，GRPO 可以一句话概括：

> **GRPO = REINFORCE + 组内均值 baseline + ratio clipping + KL 正则**

经典 REINFORCE：

$$
\nabla_\theta J = \mathbb{E}_{o \sim \pi_\theta}\big[(r(o) - b) \nabla_\theta \log \pi_\theta(o \mid q)\big]
$$

GRPO 把 baseline $b$ 设成同组均值、用 std 归一化、加上 PPO 风格的 importance sampling ratio + clip，就得到了完整的 GRPO 算法。

本质上，GRPO 是 **在 REINFORCE 和 PPO 之间找到了一个很好的平衡点**：

* 比 REINFORCE 稳定（有 clip + KL）
* 比 PPO 简单（不需要 Critic + GAE）
* 在 LLM 场景下，sequence-level reward 天然可用（Reward Model 本来就是对整个回答打分），所以 token-level 的 Critic 反而是不必要的复杂度
