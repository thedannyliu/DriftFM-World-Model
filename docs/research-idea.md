可以。這個研究不應只做成「把 DriftWorld 的 drifting loss 換成 Drift Flow Matching」，那樣方法貢獻偏弱。較完整、可驗證的研究主題應是：

# DriftFlowWorld

## Test-Time Scalable World Models with Motion-Gated NFE Advantage Distillation

核心問題是：

> 能否讓同一個 action-conditioned world model，在 **NFE=1 時保留 DriftWorld 的即時生成能力**，並在接觸、遮擋、快速運動等困難 transition 上，透過 **NFE=2/4 的 iterative transport 顯著改善動態準確度與長期 rollout**？

最適合主張的優勢不是「四步仍比 DriftWorld 一步更快」，而是：

[
\boxed{
\text{更好的 quality–latency Pareto frontier}
}
]

以及：

[
\boxed{
\text{固定 planning latency 下，更高的 policy improvement}
}
]

---

# 1. 研究動機與 DriftWorld 的可改進點

DriftWorld 的重要優勢是將 action-conditioned future video 直接生成壓縮到一次 forward；它使用目前影像、三張 history frames 與 future action chunk，預測未來一幀或一個 video chunk。它也透過 no-action negatives、DINO feature drifting、motion weighting 與 self-forcing，改善 action following 和 autoregressive rollout。([arXiv][1])

但它的 inference compute 是固定的：

[
\hat y = G_\theta(\epsilon,c)
]

不論 transition 是：

* 幾乎靜止的背景；
* 簡單平移；
* gripper 接觸物體；
* 遮擋後重新出現；
* 精密插入或抓取；

都只能使用同一次 forward。

Drift Flow Matching 則學習任意兩個 marginal times 之間的 transport map，能一次從 noise 跳到資料，也能把區間切成多段進行 iterative refinement。DFM 已在生成與 robotic policy 實驗中呈現 NFE scaling，例如部分較困難控制任務由 NFE=1 增加到 NFE=10 時有明顯成功率提升；但那是 action policy，而不是 action-conditioned video world model。([arXiv][2])

因此這篇新研究的核心 gap 是：

[
\boxed{
\text{將 DFM 的 test-time scaling 搬到 action-conditioned world modeling}
}
]

DriftWorld 已有很強的一步結果，但仍存在合理的提升空間。例如，其 Push-T full-episode rollout 的誤差高於較短的 64-frame rollout；在 RT-1 和 Language Table 上，它的 SSIM、FVD 或速度很強，但 FID 並不總是最佳。這些現象適合作為「多步 refinement 是否能改善長期、分布性與接觸品質」的研究切入點。([arXiv][1])

---

# 2. 研究主張

建議將論文主張鎖定在三項：

### Claim 1：同一模型支援可變 NFE

[
\text{NFE}=1:
\quad \text{即時、接近 DriftWorld}
]

[
\text{NFE}=2,4:
\quad \text{改善困難 transition 與長期 rollout}
]

### Claim 2：多步分支反過來改善一步分支

利用較準確的 NFE=4 rollout，選擇性地 self-distill 回 NFE=1，使一步模型不只是「保留速度」，而且在 action-sensitive motion 上可能比原始 DriftWorld 更準。

### Claim 3：固定規劃預算下更好的 policy performance

先用 NFE=1 篩選大量 action proposals，再用 NFE=4 重算 top-(K) proposals：

[
\text{coarse screening}
\rightarrow
\text{selective refinement}
\rightarrow
\text{final ranking}
]

這會比：

* 所有 proposals 都只用一步；
* 所有 proposals 都用四步；

有更好的 planning quality–cost trade-off。

---

# 3. 完全沿用 DriftWorld 的 task setup

輸入與預測問題維持不變：

[
c_t=
\left(
o_{t-3:t},
a_{t:t+H-1},
\ell
\right),
]

[
y_t=o_{t+1:t+H},
]

其中 (\ell) 是資料集提供時才使用的 language instruction。

| Dataset                   | Environment | Prediction horizon (H) | Generator space | 主要評估                                             |
| ------------------------- | ----------- | ---------------------: | --------------- | ------------------------------------------------ |
| Push-T                    | Simulation  |               4 frames | Pixel space     | 64-frame、full episode、GPC-RANK、policy evaluation |
| Robomimic Lift/Can/Square | Simulation  |               2 frames | Pixel space     | Single/two-view、full rollout、policy evaluation   |
| Bridge-V2                 | Real robot  |                1 frame | SD3 VAE latent  | 8-frame rollout、DINO/visual metrics              |
| RT-1                      | Real robot  |                1 frame | SD3 VAE latent  | 8-frame rollout、action following                 |
| Language Table            | Real robot  |                1 frame | SD3 VAE latent  | Full-video rollout、language-conditioned motion   |

DriftWorld 在所有資料集都使用 current frame 加三張歷史影像；Push-T、Robomimic 在 pixel space 訓練，較複雜的真實資料則使用 SD3 VAE latent，並在 DINOv2/v3 feature space 計算 drifting supervision。原本的 MSE、SSIM、PSNR、LPIPS、FID、FVD 與 timing 評估全部保留。([arXiv][1])

這樣能保證論文的主要比較是：

[
\text{DriftWorld objective}
\quad\text{vs.}\quad
\text{DriftFlowWorld objective}
]

而不是由資料、backbone 或 horizon 差異造成。

---

# 4. 模型架構：從 endpoint generator 改成 two-time transport

## 4.1 保留 DriftWorld backbone

保留：

* factorized spatial-temporal U-Net；
* history frames 與 noise 的 channel concatenation；
* frame-wise action FiLM；
* multi-view Robomimic 設定；
* action accentuation scale；
* SD3 VAE；
* DINOv3 feature loss；
* normalized multi-temperature drifting field。

唯一主要的 architecture 修改是加入 **time-pair embedding**：

[
e_{\text{time}}
===============

\operatorname{Embed}(\tau_s,\Delta\tau),
\qquad
\Delta\tau=\tau_e-\tau_s.
]

它和 action embedding 一樣注入每個 U-Net residual block 的 FiLM：

[
h'
==

\gamma(a,\tau_s,\Delta\tau)\odot h
+
\beta(a,\tau_s,\Delta\tau).
]

DFM 的消融顯示，以「目前時間加 step size」表示 two-time transport，通常比直接預測 target state 更穩定；mean-velocity parameterization 也較適合不同 NFE。([arXiv][2])

---

## 4.2 Conditional probability path

令 (E) 是 identity 或 SD3 VAE encoder：

[
z^+=E(y_t),
\qquad
\epsilon_i\sim\mathcal N(0,I).
]

建立線性 interpolant：

[
z_\tau^{(i)}
============

(1-\tau)\epsilon_i+\tau z^+.
]

對任意：

[
0\leq \tau_s<\tau_e\leq1,
]

模型預測 mean velocity：

[
u_\theta
\left(
z_{\tau_s},
\tau_s,
\Delta\tau,
c_t
\right).
]

transport map 為：

[
T_\theta
\left(
z_{\tau_s};
\tau_s,\tau_e,c_t
\right)
=======

z_{\tau_s}
+
\Delta\tau,
u_\theta
\left(
z_{\tau_s},
\tau_s,
\Delta\tau,
c_t
\right).
]

因此：

### 一步生成

[
\hat z_1
========

T_\theta(\epsilon;0,1,c_t).
]

### 四步生成

[
\hat z_1
========

T_\theta^{0.75\rightarrow1}
\circ
T_\theta^{0.5\rightarrow0.75}
\circ
T_\theta^{0.25\rightarrow0.5}
\circ
T_\theta^{0\rightarrow0.25}
(\epsilon).
]

DFM 的關鍵不是更換 interpolant，而是直接學習不同 marginal pairs 之間的 distribution transport，而非僅回歸一個 pointwise velocity target。([arXiv][2])

---

# 5. Conditional DFM drifting objective

DriftWorld 的每個 observation-action condition 通常只有一個 ground-truth future：

[
(c_t,y_t).
]

但可對相同 condition 抽取 (K) 個 noise：

[
\epsilon_1,\ldots,\epsilon_K.
]

形成 target-time positive empirical distribution：

[
P^+_{\tau_e,c}
==============

\frac{1}{K}
\sum_{i=1}^{K}
\delta_{
z_{\tau_e}^{(i)}
}.
]

模型的 predicted marginal 為：

[
Q_{\theta,\tau_e,c}
===================

\frac{1}{K}
\sum_{i=1}^{K}
\delta_{
\hat z_{\tau_e}^{(i)}
}.
]

令 drifting field 為：

[
D_{P,Q}(x)
==========

M_P(x)-M_Q(x),
]

其中 (M_P) 和 (M_Q) 是 kernel-weighted attraction terms。Loss 為：

[
\mathcal L_{\mathrm{DFM}}
=========================

\frac1K
\sum_i
\left|
\hat z_{\tau_e}^{(i)}
---------------------

\operatorname{sg}
\left[
\hat z_{\tau_e}^{(i)}
+
\eta
D_{P^+,Q_\theta}
\left(
\hat z_{\tau_e}^{(i)}
\right)
\right]
\right|_2^2.
]

對每一個 observation-action condition 必須獨立估計 drift field，不能把不同 history 或 action sequence 的 particles 混在一起。DFM 的 conditional formulation 特別指出，高維條件只有一個 observed target 時，仍可利用多個 source noises 建立 conditional marginal pair，這正好符合 DriftWorld 的資料結構。([arXiv][2])

初始 particle 數可以直接沿用 DriftWorld：

[
K=
\begin{cases}
8 & \text{Push-T}\
32 & \text{Robomimic}\
64 & \text{real datasets}
\end{cases}
]

DFM 的消融顯示 particles 過少會明顯破壞 one-step 與 few-step generation，因為 large-step transport 特別依賴可靠的 empirical drift estimate。([arXiv][2])

---

# 6. 保留 DriftWorld 的 action-specific design

DFM 不能直接取代 DriftWorld 所有 world-model-specific tricks。下列元件應完整保留。

## 6.1 No-action negative

仍將 repeated current frame：

[
y_{\mathrm{noact}}
==================

[o_t,\ldots,o_t]
]

加入 negative distribution，防止模型忽略 action 而單純複製背景。

在任意 intermediate time，可建立：

[
z_{\tau}^{\mathrm{noact},i}
===========================

(1-\tau)\epsilon_i
+
\tau E(y_{\mathrm{noact}}).
]

因此 action accentuation 不只作用於 endpoint，也可以作用於 intermediate marginal transport。

## 6.2 Motion-weighted spatial drifting

[
m_p
===

\left|
\phi_p(y_t)
-----------

\phi_p(o_t)
\right|_2,
]

[
w_p
===

1+\alpha\tanh(\gamma m_p).
]

gripper、物體與接觸區域有較高權重；背景則接近基礎權重。DriftWorld 的消融顯示，移除 motion weighting 後，真實機器人資料中的 gripper 容易變得近乎靜止，因此這部分不能省略。([arXiv][1])

## 6.3 Dual-space supervision

對所有 time pairs 計算：

[
\mathcal L_{\mathrm{DFM}}^{\mathrm{pixel/latent}}.
]

但 DINO semantic loss 只在：

[
\tau_e=1
\quad\text{或}\quad
\tau_e>0.9
]

時使用。

原因是 intermediate VAE latent 混有 noise，decode 後未必是自然影像；此時 DINO distance 未必具有可靠語義。完整 transport path 用 latent supervision，接近 endpoint 時再加入：

[
\mathcal L_{\mathrm{DFM}}^{\mathrm{DINO}}.
]

DriftWorld 本身也顯示 DINO feature supervision 對真實場景的 gripper sharpness 和 semantic quality 很重要。([arXiv][1])

---

# 7. 關鍵訓練 trick：Motion-Gated NFE Advantage Distillation

這是新研究最值得強調的額外貢獻。

## 7.1 問題

理論上 DFM 支援任意 NFE，但有限模型容量下：

[
\hat y_{\mathrm{NFE}=1}
\neq
\hat y_{\mathrm{NFE}=4}.
]

更多 NFE 不一定在每個 training sample 上都更好。若直接強迫一步輸出等於四步輸出，可能把四步分支的錯誤反向蒸餾給一步模型。

因此不能使用無條件 consistency loss。

---

## 7.2 一步 student 與四步 EMA teacher

使用相同 noise 和 condition：

[
\hat y_1
========

D\left(
T_\theta(\epsilon;0,1,c_t)
\right),
]

[
\hat y_4
========

D\left(
T_{\bar\theta}^{[4]}(\epsilon;c_t)
\right),
]

其中：

* (\theta)：online model；
* (\bar\theta)：EMA model；
* (D)：pixel identity 或 VAE decoder；
* teacher 不反向傳播。

---

## 7.3 先判斷四步是否真的更好

定義 motion-weighted semantic error：

[
e_N
===

\frac{
\sum_p
w_p
\left|
\phi_p(\hat y_N)
----------------

\phi_p(y_t)
\right|_1
}{
\sum_p w_p
}.
]

使用 soft advantage gate：

[
g
=

\operatorname{sg}
\left[
\sigma
\left(
\frac{
e_1-e_4-\delta
}{
T
}
\right)
\right].
]

其中：

* (e_4<e_1-\delta)：四步明顯較好，(g) 接近 1；
* 四步沒有改善：(g) 接近 0；
* (\delta)：避免蒸餾微小、可能只是 noise 的差異；
* (T)：控制 gate sharpness。

---

## 7.4 蒸餾 action-relevant semantics

[
\mathcal L_{\mathrm{MNAD}}
==========================

g
\cdot
\frac{
\sum_p
w_p
\left|
\phi_p(\hat y_1)
----------------

\operatorname{sg}
\left[
\phi_p(\hat y_4)
\right]
\right|_1
}{
\sum_p w_p
}.
]

這個 loss 有三個刻意的設計：

1. **只有四步確實比較好時才蒸餾。**
2. **只強調有運動的 gripper、物體與接觸區域。**
3. **只匹配 semantic features，而不是像素。**

第三點很重要。若直接要求：

[
\hat y_1\approx\hat y_4
]

在 pixel space 完全一致，會抑制四步 refinement 改善 texture 和細節的能力。Semantic consistency 只要求兩者對：

* 物體位置；
* gripper pose；
* 運動方向；
* 接觸結果；

保持一致，而允許高 NFE 產生更好的細節。

flow-map 的 direct map 與 composed map consistency 本身已有相關研究，因此論文不應將「consistency」本身描述為全新概念。真正的新意是：

[
\boxed{
\text{只蒸餾 multi-NFE 真正改善的 action-relevant semantic regions}
}
]

並把它用於 action-conditioned world-model rollout 與 coarse-to-fine planning。([arXiv][3])

---

# 8. NFE-mixed self-forcing

DriftWorld 的 self-forcing 第二階段會將自己生成、且 detach 的 history frames 作為下一次預測輸入，以降低 autoregressive train-test mismatch。([arXiv][1])

這裡將它改為：

[
N_{\mathrm{history}}
\sim
{1,2,4}.
]

例如：

[
P(N=1,2,4)
==========

(0.5,0.3,0.2).
]

模型有時從一步生成的 history 繼續 rollout，有時從較精細的兩步或四步 history 繼續。

這能讓同一模型適應：

* 全程 NFE=1 的快速 rollout；
* 全程 NFE=4 的高品質 rollout；
* 不同時間使用不同 NFE 的 adaptive rollout；
* coarse-to-fine planning 中產生的混合品質歷史。

這個設計尤其適合長 episode，因為 world model 真正看到的 history distribution 取決於先前使用了多少 inference steps。

---

# 9. 完整訓練目標

[
\mathcal L
==========

\mathcal L_{\mathrm{DFM}}^{\mathrm{pixel/latent}}
+
\lambda_{\mathrm{sem}}
\mathbf 1[\tau_e>0.9]
\mathcal L_{\mathrm{DFM}}^{\mathrm{DINO}}
+
\lambda_{\mathrm{MNAD}}
\mathcal L_{\mathrm{MNAD}}.
]

no-action negatives、motion weighting 與 multi-temperature aggregation直接整合進前兩個 drifting losses。

建議初始範圍：

[
\lambda_{\mathrm{sem}}=1,
]

[
\lambda_{\mathrm{MNAD}}
\in
{0.05,0.1,0.2},
]

[
\delta
\in
{0,0.02,0.05},
]

[
T
\in
{0.05,0.1,0.2}.
]

MNAD 不需要在所有 samples 上執行。可以只對：

* 20–25% minibatches；
* 每個 condition 的一個 noise particle；

執行四步 EMA teacher，控制訓練成本。

---

# 10. 建議的三階段訓練 curriculum

| 階段                    |   訓練比例 | Time-pair sampling               | Self-forcing | MNAD   |
| --------------------- | -----: | -------------------------------- | ------------ | ------ |
| A. Endpoint warmup    |  前 20% | 僅 (0\rightarrow1)                | 無            | 無      |
| B. DFM transport      | 中間 60% | 25% endpoint、75% arbitrary pairs | 無            | 後半逐步開啟 |
| C. Rollout adaptation | 最後 20% | 同上                               | NFE-mixed    | 完整開啟   |

Arbitrary pairs 建議先獨立抽兩個 logit-normal time samples，再排序為：

[
\tau_s<\tau_e.
]

DFM 的實驗中，logit-normal time-pair sampler 表現較好；endpoint replay 則是這篇 proposed method 額外加入的設計，用來避免學習 arbitrary transports 時犧牲 NFE=1 品質。([arXiv][2])

新增 time-pair FiLM layer 建議 zero-initialize，使訓練初期的網路行為接近 endpoint DriftWorld，而不會因為新增 time conditioning 突然破壞已學到的 action mapping。

---

# 11. 推論模式

## 固定 NFE

[
\begin{aligned}
N=1 &: [0,1]\
N=2 &: [0,0.5,1]\
N=4 &: [0,0.25,0.5,0.75,1]\
N=8 &: [0,0.125,\ldots,1].
\end{aligned}
]

每個 autoregressive chunk 可以選擇獨立 NFE。

---

## Coarse-to-fine GPC-RANK

DriftWorld 原本使用 GPC-RANK：policy 產生多個 action proposals，world model rollout 後由 reward model 排名。其 Push-T 實驗顯示這能提升 policy IoU，且 DriftWorld 對 proposals 的 rollout 時間顯著低於多步 world-model baselines。([arXiv][1])

新的推論演算法如下：

1. policy 產生 (M) 個 action chunks；
2. 全部以 NFE=1 rollout；
3. reward model 初步評分；
4. 保留 top-(K)；
5. 以相同 noise、NFE=4 重新 rollout top-(K)；
6. 重新評分後執行最佳 action。

總 network evaluations 為：

[
M+4K.
]

若：

[
K=0.1M,
]

則總成本為：

[
1.4M,
]

相比所有 proposals 都用 NFE=4 的：

[
4M,
]

減少 65% model evaluations。

MNAD 的功能之一，就是讓 NFE=1 和 NFE=4 的 semantic outcomes 不至於完全失配，使一步 shortlist 有較高機率保留四步下真正最佳的 proposal。

---

# 12. 評估設計

## 12.1 原本 visual metrics

所有方法都報告：

* MSE；
* SSIM；
* PSNR；
* LPIPS；
* FID；
* FVD；
* seconds per frame；
* peak GPU memory。

每個 DFM 模型報告：

[
NFE=1,2,4,8.
]

不能只報最佳 NFE，必須畫完整 quality–latency curve。

---

## 12.2 Action-Conditioned Motion Alignment

原本視覺指標容易被大面積靜態背景主導。建議新增：

[
\Delta\phi_p^{\mathrm{pred}}
============================

\phi_p(\hat y)-\phi_p(o_t),
]

[
\Delta\phi_p^{\mathrm{GT}}
==========================

\phi_p(y)-\phi_p(o_t).
]

定義：

[
\mathrm{ACMA}
=============

\frac{
\sum_p
w_p
\cos
\left(
\Delta\phi_p^{\mathrm{pred}},
\Delta\phi_p^{\mathrm{GT}}
\right)
}{
\sum_p w_p
}.
]

它直接測量預測 semantic motion 的方向是否與 ground truth 一致，特別適合驗證 MNAD 是否真的改善 action following，而不只是讓影像更銳利。

---

## 12.3 Simulator-state metrics

在 Push-T 與 Robomimic 中，可使用 simulator state 計算：

### Push-T

* block center error；
* block orientation error；
* final target IoU；
* trajectory displacement error。

### Robomimic

* end-effector pose error；
* object pose error；
* gripper-object contact F1；
* grasp/release event timing error；
* final task success prediction。

這些應作為主要 dynamics metrics，因為更低的 FVD 不必然代表更正確的物理結果。

---

## 12.4 NFE monotonicity

定義 per-sample error：

[
e_1,\ e_2,\ e_4.
]

計算：

[
\mathrm{MonoRate}
=================

P(e_4\leq e_2\leq e_1).
]

理想模型不只平均而言隨 NFE 改善，也應在大多數 samples 上呈現一致趨勢。

---

## 12.5 Long-horizon error growth

對 rollout horizon (h) 畫：

[
e(h)
]

並估計 error-growth slope：

[
s
=

\frac{e(h_2)-e(h_1)}{h_2-h_1}.
]

主要比較：

* DriftWorld NFE=1；
* DriftFlowWorld NFE=1；
* DriftFlowWorld NFE=4；
* adaptive NFE。

這比只看最終 frame 更能證明 iterative refinement 是否降低 autoregressive error accumulation。

---

## 12.6 Planning under fixed wall-clock budget

以固定預算評估，例如：

[
0.5\text{ s},\quad
1.0\text{ s},\quad
2.0\text{ s}.
]

在每個預算下允許方法自行選擇：

* proposal 數 (M)；
* refinement 數 (K)；
* NFE。

報告：

* final Push-T IoU；
* task success；
* proposals evaluated per second；
* top-(K) recall；
* reward ranking Spearman correlation。

其中：

[
\mathrm{TopKRecall}
===================

P
\left(
\arg\max_a R_{\mathrm{NFE}=4}(a)
\in
\operatorname{TopK}_{\mathrm{NFE}=1}
\right).
]

這是 coarse-to-fine planning 是否真正有效的關鍵指標。

---

# 13. Baselines

至少需要下列方法：

| Baseline                   | 作用                                      |
| -------------------------- | --------------------------------------- |
| DriftWorld                 | 原始主要 baseline                           |
| Vanilla conditional DFM    | 驗證單純換 DFM 是否已足夠                         |
| DFM + MNAD                 | 驗證主要 trick                              |
| Standard Flow Matching     | 比較 local velocity 與 arbitrary transport |
| MeanFlow                   | 比較另一種 one/few-step mean-velocity 方法     |
| MSE U-Net                  | Direct deterministic baseline           |
| GPC、VDM、LVDM               | 原始 diffusion baselines                  |
| IRASim、WorldGym、Ctrl-World | 原 DriftWorld real-world baselines       |

Flow Matching、MeanFlow 與 DFM 應盡量：

* 使用相同 U-Net；
* 相同 condition encoder；
* 相同參數量；
* 相同資料與 training steps；
* 相同 VAE/DINO；
* 分別報告相同 NFE。

否則 reviewer 很容易認為 improvement 來自 backbone 或額外訓練 compute。

---

# 14. 必要消融實驗

建議按照下列順序建立完整 ablation：

| Variant             | Conditional DFM | Motion/no-action | MNAD | Advantage gate | NFE-mixed self-forcing |
| ------------------- | --------------: | ---------------: | ---: | -------------: | ---------------------: |
| A. DriftWorld       |                 |                ✓ |      |                |                     原版 |
| B. Vanilla DFM      |               ✓ |                  |      |                |                        |
| C. DFM-World        |               ✓ |                ✓ |      |                |                        |
| D. Consistency only |               ✓ |                ✓ |    ✓ |                |                        |
| E. Advantage only   |               ✓ |                ✓ |    ✓ |              ✓ |                        |
| F. Full model       |               ✓ |                ✓ |    ✓ |              ✓ |                      ✓ |

其中最重要的比較是：

### D vs E

驗證「只在 teacher 更好時蒸餾」是否必要。

### E vs F

驗證 NFE-mixed self-forcing 是否主要改善 long-horizon，而非單幀品質。

另外應消融：

* ((\tau_s,\tau_e)) vs ((\tau_s,\Delta\tau)) embedding；
* uniform vs logit-normal time-pair sampling；
* endpoint replay probability；
* particles (K)；
* DINO endpoint threshold；
* (\lambda_{\mathrm{MNAD}})；
* teacher NFE=2 vs 4；
* EMA teacher vs online teacher；
* semantic consistency vs pixel consistency；
* (K/M) coarse-to-fine ratio。

---

# 15. 建議預先設定的成功標準

以下應被描述為 **研究目標／go-no-go criteria**，不是預先宣稱的結果。

## 一步模式

* NFE=1 latency 不高於 DriftWorld 10%；
* NFE=1 LPIPS、FVD 或 state error 不比 DriftWorld差超過 5%；
* MNAD 後的 NFE=1 ACMA 至少提升 5%。

## 多步 scaling

* NFE=4 在至少三個資料集上降低 LPIPS 或 FVD 10%；
* Push-T／Robomimic state 或 contact error 降低 15%；
* 至少 80% 的 dataset-metric combinations 隨 NFE 呈單調改善。

## Long-horizon

* full-episode error-growth slope 降低 15%；
* Push-T final block pose error 顯著小於 DriftWorld；
* Robomimic grasp/release event timing 更準。

## Planning

* 相同 wall-clock budget 下，GPC-RANK IoU 提升 (0.03)–(0.05)；
* 或在相同 IoU 下，降低至少 30% planning latency；
* top-10% proposal recall 高於 90%。

---

# 16. 主要風險與處理方式

### 風險一：DFM 降低原本的一步品質

原因是模型同時學很多 time pairs，容量被分散。

處理：

* endpoint warmup；
* 25% endpoint replay；
* zero-init time adapters；
* 提高 (0\rightarrow1) pair 的 loss weight。

### 風險二：四步 teacher 不一定比較好

處理：

* 使用 advantage gate；
* 使用 EMA teacher；
* 設定 improvement margin (\delta)；
* warmup 後才開啟 MNAD。

### 風險三：Consistency 抑制多步 refinement

處理：

* 只蒸餾 DINO semantic motion；
* 不 matching RGB pixels；
* 不蒸餾靜態背景；
* 保留高 NFE 對 texture、細節與 uncertainty 的自由度。

### 風險四：Intermediate DINO features 沒有意義

處理：

* intermediate pairs 只使用 pixel/VAE latent drift；
* DINO loss 僅用於接近 endpoint 的 states。

### 風險五：訓練成本過高

處理：

* MNAD 僅在 20–25% batches 執行；
* 每個 condition 僅選一個 particle 做 teacher rollout；
* cache ground-truth VAE 和 DINO features；
* teacher branch完全 stop-gradient。

### 風險六：視覺改善但物理結果沒改善

處理：

* 將 simulator-state、contact F1、ACMA 和 fixed-budget planning 設為 primary metrics；
* FID/FVD 只作 secondary visual metrics。

---

# 17. 最終論文故事

這篇研究最合理的論文敘事是：

1. **DriftWorld 證明 drifting 可以做到高速 one-step world modeling。**
2. **但固定 one-step 無法針對困難 contact dynamics 增加 inference compute。**
3. **Conditional DFM 將 world prediction 改成 arbitrary two-time marginal transport。**
4. **同一模型因此同時支援 NFE=1 與 iterative refinement。**
5. **Motion-Gated NFE Advantage Distillation 將多步真正改善的 action semantics 選擇性蒸餾回一步模式。**
6. **NFE-mixed self-forcing使模型能進行不同精度的 autoregressive rollout。**
7. **Coarse-to-fine GPC-RANK 在固定 planning latency 下取得更好的 action selection。**

最核心的方法式可濃縮成：

[
\boxed{
\text{DriftWorld conditioning}
+
\text{two-time DFM transport}
+
\text{selective multi-NFE self-distillation}
}
]

而主要系統結果應是：

[
\boxed{
\text{fast when possible, refined when necessary}
}
]

---

## 可直接使用的研究摘要

> 本研究提出 DriftFlowWorld，一種具測試時計算伸縮能力的 action-conditioned robot world model。現有 DriftWorld 能以單次 forward 高速生成未來觀測，但無法在困難的接觸、遮擋與長期 transition 上使用額外推論計算改善預測。我們將 Drift Flow Matching 的 two-time marginal transport 引入世界模型，使同一個模型既能以一次 transport 直接生成 future video，也能透過多個短區間 transport 進行 iterative refinement。為避免多時間訓練犧牲一步品質，我們提出 Motion-Gated NFE Advantage Distillation：只有當 EMA 多步 prediction 在 action-relevant semantic regions 上確實優於一步 prediction 時，才將其蒸餾回一步分支。我們進一步使用 NFE-mixed self-forcing，使模型適應不同精度的 autoregressive histories，並提出先以一步 rollout 篩選 action proposals、再以多步 rollout 精煉 top-(K) candidates 的 coarse-to-fine planning 方法。實驗沿用 DriftWorld 的 Push-T、Robomimic、Bridge-V2、RT-1 與 Language Table setup，評估視覺品質、動態與接觸準確度、長期 rollout、policy evaluation，以及固定 wall-clock budget 下的 inference-time policy improvement。

[1]: https://arxiv.org/html/2607.15065 "https://arxiv.org/html/2607.15065"
[2]: https://arxiv.org/html/2605.17244 "https://arxiv.org/html/2605.17244"
[3]: https://arxiv.org/html/2602.20463v1 "https://arxiv.org/html/2602.20463v1"
