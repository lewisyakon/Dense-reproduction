# Qwen3-4B ↔ Qwen3-8B Dense KV-Cache Communication 最小复现方案

## 目标

当前只复现两组有向通信：

- Qwen3-4B → Qwen3-8B
- Qwen3-8B → Qwen3-4B

每个方向单独训练一个 Adapter，不共享参数。

目标是实现：

$$
C_S(X)\rightarrow T_\theta \rightarrow \tilde C_R(X)
$$

使 Sender 的 KV Cache 能转换为 Receiver 可直接使用的 KV Cache，并同时支持：

1. **Context-Aware**：Receiver 仍能看到原始问题 \(X\)；
2. **Context-Unaware**：Receiver 看不到 \(X\)，只能依赖转换后的 Cache。

本阶段不复现：

- Qwen3-14B；
- 其余 4 个模型方向；
- compressed-sensing 分析；
- 完整 OOD 实验；
- 通用多模型 Adapter；
- 跨模型家族通信。

---

## 1. 模型配置

准备两个基础模型：

- Qwen3-4B
- Qwen3-8B

需要分别训练：

```text
adapter_4B_to_8B
adapter_8B_to_4B
```

不要假设两个方向可以共享同一个 Adapter，因为它们分别学习：

$$
C_{4B}\rightarrow C_{8B}
$$

以及：

$$
C_{8B}\rightarrow C_{4B}
$$

---

## 2. Adapter 总体流程

对于 Sender Cache：

$$
C_S
$$

执行：

$$
C_S
\rightarrow
\text{RemoveRoPE}
\rightarrow
\text{Layer Alignment}
\rightarrow
\text{KV-Group Alignment}
\rightarrow
\text{Per-Group K/V MLP}
\rightarrow
\text{Learnable Gate}
\rightarrow
\text{Add Receiver RoPE}
\rightarrow
\tilde C_R
$$

最终得到 Receiver-compatible KV Cache：

$$
\tilde C_R=T_\theta(C_S)
$$

---

## 3. Layer Alignment

设 Sender 和 Receiver 层数分别为：

$$
L_S,\quad L_R
$$

Receiver 第 \(l\) 层对应 Sender：

$$
a(l)
=
\operatorname{round}
\left(
\frac{l(L_S-1)}
{L_R-1}
\right)
$$

使用固定的单调深度映射，不训练额外的 layer router。

---

## 4. KV-Group Alignment

对于 Receiver 的每个 KV group \(g\)，定义 Sender group：

$$
\pi_l(g)
$$

如果 Qwen3-4B 和 Qwen3-8B 的 KV-group layout 一致，优先直接使用：

$$
\pi_l(g)=g
$$

即同编号 group 对齐。

论文没有明确提供可学习 group router，因此当前最小复现不要额外设计 learnable routing。

---

## 5. Per-Group K/V Transformation

对每个 Receiver：

$$
(l,g)
$$

分别建立：

$$
T_K^{(l,g)}
$$

和：

$$
T_V^{(l,g)}
$$

Key 和 Value 不共享参数。

每个 transformation 使用两层 MLP：

$$
d_S
\rightarrow
16d_S
\rightarrow
d_R
$$

中间使用 GELU。

形式为：

$$
T_{\star}^{(l,g)}(z)
=
W_{\star,2}^{(l,g)}
\operatorname{GELU}
\left(
W_{\star,1}^{(l,g)}z+b_{\star,1}^{(l,g)}
\right)
+b_{\star,2}^{(l,g)}
$$

其中：

$$
\star\in\{K,V\}
$$

该 MLP 对每个 token 的 K/V vector 独立应用。

---

## 6. Learnable Gate

为每个 Receiver layer/group 设置：

$$
\gamma^{(l,g)}\in[0,1]
$$

输出：

$$
K_R'^{(l,g)}
=
\gamma^{(l,g)}
T_K^{(l,g)}
\left(
\hat K_S^{(a(l),\pi_l(g))}
\right)
$$

$$
V_R'^{(l,g)}
=
\gamma^{(l,g)}
T_V^{(l,g)}
\left(
\hat V_S^{(a(l),\pi_l(g))}
\right)
$$

Gate 与 MLP 一起训练。

不要为 Context-Aware 和 Context-Unaware 分别使用不同 gate。

---

## 7. RoPE 处理

跨模型转换前：

$$
\hat C_S
=
\operatorname{RemoveRoPE}_S(C_S)
$$

完成内容转换后：

$$
\tilde C_R
=
\operatorname{AddRoPE}_R(C_R')
$$

目标是先对齐内容表示，再恢复 Receiver 自己的位置编码。

---

# 8. Phase I：Cache Reconstruction

## 训练目标

让 Adapter 学会：

$$
T_\theta(C_S(X))
\approx
C_R(X)
$$

即把 Sender latent representation 转换到 Receiver-native latent space。

---

## 单个训练样本流程

对于同一个输入 \(X\)：

### Sender

$$
X
\rightarrow A_S
\rightarrow C_S(X)
$$

### Receiver

$$
X
\rightarrow A_R
\rightarrow C_R(X)
$$

### Adapter

$$
\tilde C_R(X)
=
T_\theta(C_S(X))
$$

### Reconstruction Loss

$$
\mathcal L_{\mathrm{rec}}
=
\sum_{l,g}
\left\|
\tilde K_R^{(l,g)}
-
K_R^{(l,g)}
\right\|_2^2
+
\left\|
\tilde V_R^{(l,g)}
-
V_R^{(l,g)}
\right\|_2^2
$$

Phase I 训练：

- Key MLP；
- Value MLP；
- learnable gate。

不训练：

- Sender；
- Receiver；
- 固定 layer mapping；
- 固定 group mapping。

训练完成后保存：

```text
phase1_4B_to_8B.pt
phase1_8B_to_4B.pt
```

---

# 9. Phase II：Generation-Oriented Training

Phase I 只保证：

$$
\tilde C_R\approx C_R
$$

但不保证 Receiver 一定能稳定地利用转换后的 Cache 得到正确答案。

Phase II 从对应的 Phase-I checkpoint 初始化，继续训练同一个 Adapter。

---

## 10. Receiver-Self Guided Trace

对于：

$$
4B\rightarrow8B
$$

使用 **Qwen3-8B Receiver 自己生成的 reasoning trace**。

对于：

$$
8B\rightarrow4B
$$

使用 **Qwen3-4B Receiver 自己生成的 reasoning trace**。

给定：

$$
(X,y_{\mathrm{gold}})
$$

让 Receiver 同时看到：

- 原始问题；
- ground-truth answer；

然后生成一条能推导到正确答案的 step-by-step reasoning trace。

例如保存为：

```json
{
  "question": "...",
  "gold_answer": "...",
  "trace": "...",
  "receiver": "Qwen3-8B"
}
```

不要使用 Sender 的 reasoning trace 作为默认监督目标。

---

## 11. Phase II 两种训练模式

同一条训练样本随机进入两种模式。

### Context-Aware

概率：

$$
0.5
$$

Receiver 输入：

$$
X_R=X
$$

优化：

$$
P(y\mid \tilde C_R(X),X)
$$

---

### Context-Unaware

概率：

$$
0.5
$$

Receiver 输入：

$$
X_R=\varnothing
$$

优化：

$$
P(y\mid \tilde C_R(X))
$$

两种模式：

- 使用同一个 Adapter；
- 使用同一条 target reasoning trace；
- 只改变 Receiver 是否直接看到原始问题。

Generation loss：

$$
\mathcal L_{\mathrm{gen}}
=
-\sum_t
\log
P_{A_R}
\left(
y_t
\mid
y_{<t},
\tilde C_R(X),
X_R
\right)
$$

论文主实验 Phase II 训练 2000 steps，可先以此为复现目标。

---

# 12. 第一版训练数据

为了先验证方法是否跑通，只使用论文中的三类训练任务：

- GSM8K
- MATH algebra subset
- ARC-Challenge

第一版不必立即加入：

- MMLU-Redux
- MedQA
- OpenBookQA

先证明 4B↔8B 双向通信在训练域任务上可用。

---

# 13. 推理方式

## 13.1 Context-Aware

流程：

$$
X
\rightarrow A_S
\rightarrow C_S
\rightarrow T_\theta
\rightarrow \tilde C_R
$$

同时 Receiver 自己仍看到 \(X\)。

最终：

$$
P(y\mid X,\tilde C_R)
$$

重点验证：

- transformed cache 能否提升 Receiver；
- 是否优于 Receiver-only；
- 是否接近或超过 T2T。

---

## 13.2 Context-Unaware

只有 Sender 看到问题：

$$
X
\rightarrow A_S
\rightarrow C_S
\rightarrow T_\theta
\rightarrow \tilde C_R
$$

Receiver 不再看到 \(X\)，仅使用：

$$
\tilde C_R
$$

生成：

$$
P(y\mid\tilde C_R)
$$

这是当前复现中最重要的正确性测试。

如果该模式能维持较高准确率，说明 Adapter 不只是提供 reasoning steering，而确实传递了较完整的上下文信息。

---

# 14. 第一阶段实验只比较四种方法

## Receiver-only

Receiver 单独回答问题。

## T2T

Sender 生成文本 reasoning，Receiver 重新读取后回答。

## Ours：Context-Aware

Receiver 同时得到：

$$
X+\tilde C_R
$$

## Ours：Context-Unaware

Receiver 只得到：

$$
\tilde C_R
$$

第一轮可以暂时不复现 C2C，先验证本文方法自身是否正确运行。

---

# 15. 最小评估指标

对两个方向分别报告：

```text
4B → 8B
8B → 4B
```

至少测：

- GSM8K Accuracy
- MATH-500 Accuracy
- ARC-C Accuracy

以及：

- Sender prefill tokens
- Sender decode tokens
- Receiver prefill tokens
- Receiver decode tokens
- Estimated TFLOPs

TFLOPs 使用论文近似：

$$
\mathrm{FLOPs}
\approx
2N_S
(l_{\mathrm{pre}}^S+l_{\mathrm{dec}}^S)
+
2N_R
(l_{\mathrm{pre}}^R+l_{\mathrm{dec}}^R)
$$

Ours 中应满足：

$$
l_{\mathrm{dec}}^S=0
$$

因为 Sender 不执行自回归 reasoning，只做 prefill 并传递 Cache。

---

# 16. 第一轮复现成功标准

## 4B → 8B

Phase I 后：

- transformed KV Cache 与 8B native Cache 的 reconstruction loss 明显下降；
- Receiver 能正常消费 transformed cache，不出现维度/RoPE/position 错误。

Phase II 后：

- Context-Aware 能正常生成正确格式答案；
- Context-Unaware 在不输入原始问题时仍显著高于随机水平；
- 性能明显优于未经训练的直接 Cache 注入。

---

## 8B → 4B

同样要求：

- Phase I reconstruction 收敛；
- Phase II generation 正常；
- Context-Unaware 能仅依赖 transformed cache 工作。

---

# 17. 暂时不要做的扩展

当前阶段不要增加以下复杂度：

1. 14B 模型；
2. 6 个完整方向；
3. learnable group router；
4. 多 Sender 共享 Adapter；
5. 一个通用双向 Adapter；
6. Qwen → Llama 等跨家族通信；
7. Cache compression；
8. sparse head pruning；
9. compressed-sensing 分析；
10. 完整 OOD benchmark；
11. 网络传输优化；
12. Adapter 参数压缩。

先确保最基础的：

$$
\boxed{
4B\rightarrow8B
}
$$

和：

$$
\boxed{
8B\rightarrow4B
}
$$

两条通信链路能够独立训练并运行。

---

# 18. 推荐实现顺序

## Step 1

先完成：

```text
4B → 8B Phase I
```

验证：

$$
C_{4B}\rightarrow\tilde C_{8B}\approx C_{8B}
$$

---

## Step 2

完成：

```text
8B → 4B Phase I
```

验证：

$$
C_{8B}\rightarrow\tilde C_{4B}\approx C_{4B}
$$

---

## Step 3

实现 transformed Cache 注入 Receiver 的推理接口。

优先跑通 Context-Unaware：

$$
P(y\mid\tilde C_R)
$$

确保 Receiver 能仅依赖 transformed cache 生成。

---

## Step 4

生成两个方向各自的 Receiver-self guided traces。

---

## Step 5

分别进行两个 Adapter 的 Phase II：

```text
4B → 8B
8B → 4B
```

训练时：

```text
50% Context-Aware
50% Context-Unaware
```

---

## Step 6

先在：

```text
GSM8K
MATH-500
ARC-C
```

上完成双向评估。

---

## Step 7

在双向复现稳定后，再考虑：

- C2C baseline；
- OOD 数据集；
- 14B；
- compressed sensing；
- 通信带宽优化。

---

# 19. 当前最小目标

本阶段最终只需要证明：

$$
\boxed{
Qwen3\text{-}4B
\leftrightarrow
Qwen3\text{-}8B
}
$$

之间可以通过两个独立 Dense KV-Cache Adapter 实现双向 latent communication。

每个方向都应完成：

$$
\boxed{
\text{Phase I: Cache Reconstruction}
}
$$

以及：

$$
\boxed{
\text{Phase II: Generation-Oriented Training}
}
$$

并同时验证：

$$
\boxed{
\text{Context-Aware}
}
$$

和：

$$
\boxed{
\text{Context-Unaware}
}
$$

两种工作模式。

第一阶段不要追求完整论文复现，先把：

> **双向 Cache 转换 → Receiver 正常消费 → Phase I 收敛 → Phase II 能正确生成**

这条最小闭环跑通。
