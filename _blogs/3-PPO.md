---
title: "Proximal Policy Optimization (PPO)"
date: 2026-02-11
---
Introduction of Proximal Policy Optimization (PPO)


把 **PPO** 讲清楚： **Advantage A 怎么来** 、 **GAE 怎么算** 、以及  **PPO 的 loss 由哪些项组成、各自起什么作用** 。默认已经知道基本的 Actor-Critic 框架（策略 $\pi_\theta$、价值函数 $V_\phi$）。

---

## 1) PPO在优化什么：从 policy gradient 到 "稳定更新"

**目标**：最大化期望回报

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}\Big[\sum_{t}\gamma^t r_t\Big]
$$

$r_t$是环境奖励，$\gamma$是折扣因子。

经典 policy gradient：

$$
\nabla_\theta J(\theta)=\mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\big]
$$

其中核心就是  **Advantage $A_t$** ：它告诉你"这一步动作比平均水平好多少"，用来给梯度定方向和权重。

问题：直接用这个做 SGD 很容易  **策略更新太猛** ，导致性能崩（policy collapse / KL 爆炸）。
TRPO 用约束优化（KL 约束）稳定更新，PPO 用更简单的 **clip surrogate** 来近似达到"别走太远"。

---

## 2) Advantage $A_t$：到底是什么，为什么能降低方差

### 2.1 定义（最正统）

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

* $Q$：执行 $a_t$ 之后的期望回报
* $V$：在 $s_t$ 下平均期望回报（baseline）: **$V(s_t) = \mathbb{E}_{a_t \sim \pi}\big[Q(s_t, a_t)\big]$**

所以 $A>0$ 表示这个动作比平均好，应当提高 $\pi(a_t|s_t)$；反之降低。

### 2.2 实际训练里怎么估计

你没有真实 $Q^\pi$，所以用采样回报（return）或 TD 估计来近似。最常见：

* **Monte Carlo Return** ：方差大但无偏
* **TD(1-step)** ：偏差更大但方差小
* **GAE** ：在二者之间做平衡（最常用）

---

## 3) GAE：Advantage 的"偏差-方差折中器"

### 3.1 先定义 TD 误差（delta）

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

把 $V(s_t)$ 理解成"你站在 $s_t$ 时，对未来总回报的**预期**"。然后你真的走了一步，拿到了实际奖励 $r_t$，到达了 $s_{t+1}$。这时候你可以用一个更新后的估计来评价"未来总回报到底该是多少"：

$$
\underbrace{r_t}_{\text{真实拿到的}} + \underbrace{\gamma V(s_{t+1})}_{\text{到了新状态后，对剩余未来的预期}}
$$

这就是所谓的 **TD target**（一步 TD 目标）。所以 $\delta_t$ 就是：

$$
\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{走了一步后的"事后估计"}} - \underbrace{V(s_t)}_{\text{走之前的"事前预期"}}
$$

* $\delta_t > 0$：实际比预期好 —— $V(s_t)$ 低估了
* $\delta_t < 0$：实际比预期差 —— $V(s_t)$ 高估了
* $\delta_t \approx 0$：critic 估得很准

本质上 $\delta_t$ 就是 **1-step 版本的 Advantage**（$\lambda=0$ 时 $A_t = \delta_t$）。GAE 做的事就是把很多步的 $\delta$ 加权求和，得到更稳定的 Advantage 估计。

### 3.2 GAE 的核心公式

$$
A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l \,\delta_{t+l}
$$

直观理解：

* 把未来很多步的 TD 残差按 $(\gamma\lambda)^l$ 衰减加权求和
* $\lambda$ 控制你更信任长回报还是短回报

### 3.3 $\lambda$ 的含义（很重要）

* $\lambda=0$：只用一步 TD：$A_t=\delta_t$（低方差，高偏差）
* $\lambda\to 1$：逼近 MC（低偏差，高方差）
* 实践里 $\lambda \in [0.9,0.97]$ 常见

### 3.4 你在实现里会用的"反向递推写法"

对一个 rollout（长度 T，遇到终止 done 要截断）：

$$
A_t = \delta_t + \gamma\lambda(1-done_{t})A_{t+1}
$$

从 $t=T-1$ 往前扫一遍即可。

### 3.5 Return（给 value loss 用）怎么来

Critic 的训练目标是让 $V_\phi(s_t)$ 尽可能接近"真实回报"。问题是你没有真实回报，需要一个估计量 $\hat{R}_t$ 来做回归目标。

**最常见做法（GAE 顺带就算出来了）**：

$$
\hat{R}_t = A_t^{\text{GAE}} + V(s_t)
$$

**为什么这个公式成立？** 回忆 Advantage 的定义：

$$
A_t = Q(s_t,a_t) - V(s_t) \quad\Longrightarrow\quad Q(s_t,a_t) = A_t + V(s_t)
$$

而 $Q(s_t,a_t)$ 本身就是"在 $s_t$ 执行 $a_t$ 后的期望回报"，所以 $\hat{R}_t = A_t + V(s_t)$ 就是用 GAE 估出来的 $Q$ 值。

**但等一下，$V$ 估计的是所有动作的平均回报，拿某个具体动作的 $Q$ 当 $V$ 的 target 不是错了吗？**

并没有。关键在于 **采样期望**：

$$
V(s_t) = \mathbb{E}_{a_t \sim \pi}\big[Q(s_t, a_t)\big]
$$

每次 rollout 采到的 $a_t$ 是从 $\pi$ 里抽出来的，所以这个样本的 $\hat{R}_t$（即 $Q(s_t, a_t)$）虽然是某一个动作的回报，但 **在期望意义下它就等于 $V(s_t)$**。这和监督学习一样：你要拟合 $\mathbb{E}[Y|X]$，手里只有单个样本 $y_i$，用 MSE loss 训练，模型最终收敛到条件期望：

$$
\min_\phi \;\mathbb{E}\big[(V_\phi(s_t) - \hat{R}_t)^2\big] \quad\Longrightarrow\quad V_\phi \to \mathbb{E}[\hat{R}_t \mid s_t] = V^\pi(s_t)
$$

每个 $\hat{R}_t$ 是某一次动作的回报（有噪声），但大量样本平均下来，$V_\phi$ 会收敛到真正的 $V$。

**其他算 Return 的方式（了解即可）**：

| 方法          | 公式                                                                   | 特点                      |
| ------------- | ---------------------------------------------------------------------- | ------------------------- |
| MC Return     | $\hat{R}_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k}$                     | 无偏，方差大              |
| 1-step TD     | $\hat{R}_t = r_t + \gamma V(s_{t+1})$                                | 偏差大，方差小            |
| n-step Return | $\hat{R}_t = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n})$ | 居中                      |
| GAE-based     | $\hat{R}_t = A_t^{\text{GAE}} + V(s_t)$                              | 同时复用 GAE 结果，最常用 |

**实践要点**：

* $\hat{R}_t$ 里的 $V(s_t)$ 是 **采样时** 的 value（即 $V_{\text{old}}$），不是当前正在更新的 value。训练中 $V_\phi$ 在变，但目标 $\hat{R}_t$ 在一轮 rollout 内是固定的
* 这也是为什么代码里通常在 rollout 阶段就把 $A_t$ 和 $\hat{R}_t$ 都算好、存进 buffer，后面多个 epoch 复用

---

## 4) PPO 的核心：ratio + clip 的 surrogate objective

### 4.1 重要性采样比率（因为用旧策略采样）

rollout 是用旧策略 $\pi_{\theta_\text{old}}$ 收集的，但你在更新 $\theta$。于是用：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
$$

### 4.2 不 clip 的 surrogate（会很不稳定）

$$
L^{\text{PG}}(\theta)=\mathbb{E}\big[r_t(\theta)A_t\big]
$$

如果 $A_t>0$ 且 $r_t$ 被推得很大，就会疯狂增大概率，更新过猛。

### 4.3 PPO-Clip 的关键

$$
L^{\text{CLIP}}(\theta)=
\mathbb{E}\Big[
\min\big(
r_t(\theta)A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon) \cdot A_t
\big)
\Big]
$$

**分情况直觉**：

* 若 $A_t>0$：你想增大 $r_t$，但一旦 $r_t>1+\epsilon$，就用 clip 后的上限，防止过猛。
* 若 $A_t<0$：你想减小 $r_t$，但一旦 $r_t<1-\epsilon$，clip 把下降幅度限制住。

这就是 PPO 稳定更新的灵魂： **对每个样本的"改动幅度"设软上限** 。

---

## 5) PPO 的 loss 组成：Policy / Value / Entropy（以及常见扩展）

一个典型 PPO 总损失（要最小化）写成：

$$
\mathcal{L}=
-\,L^{\text{CLIP}}(\theta)
+\,c_v\,\mathcal{L}_V(\phi)
-\,c_e\,\mathcal{H}(\pi_\theta)
$$

注意：很多论文/代码把 policy objective 写成要最大化，所以放到 loss 里会带负号。

下面逐项解释：

---

### 5.1 Policy loss（Actor）：让好动作更可能、坏动作更不可能，但别走太远

$$
\mathcal{L}_{\text{policy}} = -L^{\text{CLIP}}
$$

**作用**：

* 用 $A_t$ 指导方向（credit assignment）
* 用 clip 控制更新幅度（稳定性）

**实践细节**：

* $A_t$ 通常会做标准化：$A \leftarrow (A-\mu)/(\sigma+1e{-8})$，显著稳定训练

---

### 5.2 Value loss（Critic）：拟合回报（baseline 质量直接决定 PPO 效果）

最简单：

$$
\mathcal{L}_V = \mathbb{E}\big[(V_\phi(s_t)-\hat{R}_t)^2\big]
$$

**作用**：

* 价值网络越准，Advantage 方差越小，policy 更新越稳
* critic 很烂时，PPO 也会变得"像在瞎更新"

**PPO 常见改良：Value clipping**
和 policy 类似，也限制 value 更新幅度：

$$
V_{\text{clip}} = V_{\text{old}} + \text{clip}(V - V_{\text{old}},-\epsilon_v,\epsilon_v)
$$

$$
\mathcal{L}_V=\mathbb{E}\big[\max((V-\hat{R})^2,\;(V_{\text{clip}}-\hat{R})^2)\big]
$$

**作用**：防止 critic 一步跳太远导致目标漂移、训练不稳。

---

### 5.3 Entropy bonus：防止过早变"确定性策略"（探索/多样性）

熵：

$$
\mathcal{H}(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log \pi(a|s)
$$

（连续动作是高斯熵）

加到目标里等价于鼓励更大的动作分布。

$$
\mathcal{L}_{\text{entropy}}=-c_e\,\mathcal{H}
$$

**作用**：

* 抵抗 mode collapse（策略太快收缩）
* 在稀疏奖励任务里尤其关键

---

### 5.4 常见额外项：KL penalty / early stopping

虽然 PPO-clip 不强制 KL，但工程里常加：

* 监控 $\mathrm{KL}(\pi_{\text{old}}\|\pi)$，超过阈值就 early stop 当前 epoch
* 或加 KL penalty：$+\beta \,\mathrm{KL}$

**作用**：进一步保证"别走太远"。

---

## 6) PPO 的训练流程（你对齐代码就懂）

1. 用当前策略 $\pi_{\theta_{\text{old}}}$ 跑环境收集 rollout：$(s_t,a_t,r_t,done_t,\log\pi_{\text{old}},V_{\text{old}})$
2. 用 critic 计算 $V(s_t)$，算 TD 残差 $\delta_t$
3. 用 GAE 反向递推得到 $A_t$，再得 $\hat{R}_t=A_t+V(s_t)$
4. 多个 epoch 在同一批 rollout 上做 SGD：
   * 计算 ratio $r_t=\exp(\log\pi-\log\pi_{\text{old}})$
   * policy clip loss
   * value loss（可 value clip）
   * entropy bonus
   * 合成总 loss，反向传播
5. 更新 $\theta_{\text{old}}\leftarrow \theta$，进入下一轮采样

---

## 7) 一些"细节坑点"（非常影响能不能跑稳）

* **GAE 要正确处理 done** ：终止状态后不应 bootstrap 到下一状态
* **Advantage 标准化** ：几乎是标配
* **学习率 + epoch 数 + batch size** ：PPO 是 on-policy，但会对同一批数据做多轮更新；更新太多轮会过拟合这批数据（ratio/kl 会飙）
* **clip $\epsilon$** ：太小学不动，太大不稳。常见 0.1～0.3
* **reward scaling / normalization** ：连续控制里很常见
* **critic 过强或过弱都不行** ：过弱 A 噪声大；过强也可能让策略更新变"保守但错误"（取决于偏差）
