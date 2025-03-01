### **策略梯度方法（Policy Gradient Method）基礎知識**

策略梯度方法（Policy Gradient, PG）是**強化學習（Reinforcement Learning, RL）**中的一類方法，用於**直接學習最優策略（Optimal Policy）**，使智能體（Agent）在環境（Environment）中學習如何最大化累積獎勵（Reward）。  
與基於值函數的方法（如 Q-learning, DDPG）不同，策略梯度方法直接對策略（Policy）進行參數化並使用梯度上升來優化。

---

## **1. 策略表示**

在策略梯度方法中，策略（Policy）通常是**參數化的函數**，用 πθ(a∣s)\pi_\theta(a|s)πθ​(a∣s) 表示，表示在狀態 sss 下選擇動作 aaa 的機率：

πθ(a∣s)=P(a∣s;θ)\pi_\theta(a|s) = P(a | s; \theta)πθ​(a∣s)=P(a∣s;θ)

其中：

- θ\thetaθ 是策略的可調整參數（例如神經網路權重）。
- 若策略是**確定性策略**，則輸出為動作： a=πθ(s)a = \pi_\theta(s)a=πθ​(s)
- 若策略是**隨機策略**，則輸出為動作的機率分佈： a∼πθ(⋅∣s)a \sim \pi_\theta(\cdot|s)a∼πθ​(⋅∣s)

策略的目標是**最大化累積回報**，即：

J(θ)=Eτ∼πθ[R(τ)]J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]J(θ)=Eτ∼πθ​​[R(τ)]

其中：

- τ\tauτ 代表一條軌跡（Trajectory）τ=(s0,a0,r0,s1,a1,r1,… )\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)τ=(s0​,a0​,r0​,s1​,a1​,r1​,…)。
- R(τ)R(\tau)R(τ) 是該軌跡的累積獎勵（Return）。
- 目標是調整 θ\thetaθ 來最大化 J(θ)J(\theta)J(θ)。

---

## **2. 策略梯度定理（Policy Gradient Theorem）**

為了最大化 J(θ)J(\theta)J(θ)，我們可以對策略參數 θ\thetaθ 進行梯度上升：

∇θJ(θ)=Eτ∼πθ[∑t=0T∇θlog⁡πθ(at∣st)Rt]\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R_t \right]∇θ​J(θ)=Eτ∼πθ​​[t=0∑T​∇θ​logπθ​(at​∣st​)Rt​]

這稱為**策略梯度定理（Policy Gradient Theorem）**，表示：

- **對數梯度估計（Log Derivative Trick）**：將策略的梯度表示為對數的梯度： ∇θπθ(a∣s)=πθ(a∣s)∇θlog⁡πθ(a∣s)\nabla_\theta \pi_\theta(a | s) = \pi_\theta(a | s) \nabla_\theta \log \pi_\theta(a | s)∇θ​πθ​(a∣s)=πθ​(a∣s)∇θ​logπθ​(a∣s)
- **無需顯式建模環境**：不需要知道環境的轉移概率 P(s′∣s,a)P(s' | s, a)P(s′∣s,a)。
- **梯度估計依賴累積回報 RtR_tRt​**，即高回報的行為應該被增強。

---

## **3. 策略梯度方法的主要算法**

### **(1) REINFORCE（蒙特卡洛策略梯度）**

REINFORCE 是最基本的策略梯度方法，步驟如下：

1. **收集一條或多條完整的軌跡** τ=(s0,a0,r0,… )\tau = (s_0, a_0, r_0, \dots)τ=(s0​,a0​,r0​,…)。
2. **計算累積回報**： Rt=∑k=tTγk−trkR_t = \sum_{k=t}^{T} \gamma^{k-t} r_kRt​=k=t∑T​γk−trk​ 其中 γ\gammaγ 是折扣因子。
3. **更新策略參數 θ\thetaθ**： θ←θ+α∑t=0T∇θlog⁡πθ(at∣st)Rt\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R_tθ←θ+αt=0∑T​∇θ​logπθ​(at​∣st​)Rt​
    - α\alphaα 是學習率（Learning Rate）。
    - RtR_tRt​ 作為**回報權重**，高回報的行為會增加選擇概率。

#### **REINFORCE 缺點**

- **高方差（Variance 高）**：回報 RtR_tRt​ 直接影響梯度更新，導致收斂慢。
- **不能處理連續動作空間**：標準 REINFORCE 適用於離散動作空間。

---

### **(2) Actor-Critic（行為者-評論者）**

為了降低 REINFORCE 的方差，**Actor-Critic（AC）** 方法引入了**值函數 V(s)V(s)V(s)**，將策略（Actor）和評估（Critic）結合：

- **Actor（行為者）** 負責選擇動作，更新策略： θ←θ+α∇θlog⁡πθ(at∣st)At\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) A_tθ←θ+α∇θ​logπθ​(at​∣st​)At​
- **Critic（評論者）** 負責估算**狀態值函數** V(s)V(s)V(s)，用來減少策略梯度的方差： At=Rt−V(st)A_t = R_t - V(s_t)At​=Rt​−V(st​) 其中：
    - AtA_tAt​ 是**優勢函數（Advantage Function）**，用來衡量某個動作相對於平均水準的好壞。
    - V(st)V(s_t)V(st​) 透過一個**值網路（Value Network）**學習。

#### **Actor-Critic 優勢**

- **降低方差**，加快收斂。
- **適用於連續動作空間**。

---

## **4. 進階變種**

### **(1) Trust Region Policy Optimization（TRPO）**

- 為了防止策略更新過大，TRPO 限制策略更新的 KL 散度： DKL(πθold∣∣πθnew)≤δD_{\text{KL}}(\pi_{\theta_{\text{old}}} || \pi_{\theta_{\text{new}}}) \leq \deltaDKL​(πθold​​∣∣πθnew​​)≤δ
- 可以保證每次更新策略不會劇烈變動，提高穩定性。

### **(2) Proximal Policy Optimization（PPO）**

- TRPO 計算成本高，PPO 透過 **剪裁（Clipping）** 來限制策略變化： L(θ)=min⁡(rtAt,clip(rt,1−ϵ,1+ϵ)At)L(\theta) = \min\left( r_t A_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) A_t \right)L(θ)=min(rt​At​,clip(rt​,1−ϵ,1+ϵ)At​) 其中 rt=πθ(at∣st)πθold(at∣st)r_t = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}rt​=πθold​​(at​∣st​)πθ​(at​∣st​)​ 是新舊策略的比值。
- PPO 計算簡單，效果好，是目前應用最廣泛的策略梯度方法。

---

## **5. 策略梯度 vs. 值函數方法**

|方法|特點|優勢|缺點|
|---|---|---|---|
|**Q-learning / DQN**|基於值函數|適合離散動作空間|連續動作需離散化，收斂不穩定|
|**REINFORCE**|基於策略|簡單易實現，適用於隨機策略|高方差，收斂慢|
|**Actor-Critic**|策略 + 值函數|降低方差，適用於連續動作|仍需學習兩個網路|
|**PPO**|近端策略優化|計算效率高，穩定|需要調整剪裁參數|

---

## **6. 總結**

- **策略梯度方法直接學習策略 πθ\pi_\thetaπθ​**，不同於 Q-learning 這類基於值函數的方法。
- **REINFORCE 是最基本的策略梯度方法**，但方差大，收斂慢。
- **Actor-Critic 方法結合值函數，減少梯度估計的方差**，提升學習效率。
- **PPO 是目前最流行的策略梯度方法**，計算穩定，效果好，廣泛應用於深度強化學習。

這些方法廣泛應用於自動駕駛、機器人控制、遊戲 AI 等領域。