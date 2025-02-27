## **梯度爆炸（Gradient Explosion）的基礎知識與考點**

### **1. 什麼是梯度爆炸？**

梯度爆炸（Gradient Explosion）是深度學習訓練中的一個常見問題，指的是在 **反向傳播（Backpropagation）** 過程中，隨著網路層數增加，梯度在鏈式法則（Chain Rule）下不斷累積，導致梯度值變得極大，使得權重更新過大，最終導致 **模型發散（diverge）**，無法正常收斂。

梯度爆炸的問題特別常見於：

- **深層神經網路（DNN）**
- **循環神經網路（RNN）**
- **卷積神經網路（CNN）**

---

### **2. 梯度爆炸的數學描述**

在反向傳播時，梯度透過 **鏈式法則（Chain Rule）** 進行傳遞：

∂L∂W=∂L∂an⋅∂an∂an−1⋯∂a2∂a1⋅∂a1∂W\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial W}∂W∂L​=∂an​∂L​⋅∂an−1​∂an​​⋯∂a1​∂a2​​⋅∂W∂a1​​

當網絡層數增加時，梯度是多個權重矩陣的連乘，若這些矩陣的數值大於 1，則梯度會呈 **指數級增長**，導致梯度爆炸。

例如，對於某一層的權重梯度：

![[梯度爆炸 chain rule.png]]
當 WWW 的特徵值大於 1，則 gtg_tgt​ 會呈指數級增長，導致梯度爆炸。

---

### **3. 梯度爆炸的影響**

- **權重更新過大**：模型的參數變動過快，無法有效收斂。
- **Loss 震盪或發散**：損失函數可能突然變得極大或無法降低。
- **數值溢出（NaN）**：由於梯度過大，數值可能超出可計算範圍，導致計算結果變為 NaN 或 Inf。

---

### **4. 如何防止梯度爆炸？**

#### **(1) 梯度裁剪（Gradient Clipping）**

梯度裁剪（Gradient Clipping）是一種控制梯度大小的技術，透過設置閾值來限制梯度的最大值，防止梯度爆炸。

常見的梯度裁剪方式：

![[L2 范數裁剪.png]]

在 PyTorch 中：

python

複製編輯

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

在 TensorFlow 中：

python

複製編輯

`optimizer.apply_gradients(zip(gradients, variables), clipnorm=1.0)`

---

#### **(2) 權重初始化（Weight Initialization）**

合適的權重初始化可以避免梯度過大，例如：

- ![[權重初始化.png]]

在 PyTorch 中：

python

複製編輯

`torch.nn.init.xavier_uniform_(model.layer.weight)`

在 TensorFlow/Keras 中：

python

複製編輯

`initializer = tf.keras.initializers.HeNormal()`

---

#### **(3) 使用適當的激活函數**

某些激活函數（如 Sigmoid、Tanh）容易造成梯度爆炸，因此可使用：

- **ReLU（Rectified Linear Unit）** f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x)
- **Leaky ReLU、ELU、GELU** 來避免梯度問題。

---

#### **(4) 使用適當的學習率**

學習率過大可能導致梯度爆炸，因此可以：

- 使用 **學習率調整策略**（如 Learning Rate Decay、Warm-up）
- 使用 **自適應學習率優化器**（如 Adam、RMSprop）

---

### **5. 考點整理**

|**考點**|**內容**|
|---|---|
|**定義**|梯度在反向傳播中指數級增長，導致權重更新過大，使模型無法收斂|
|**數學原理**|由鏈式法則導致梯度的矩陣乘積快速增長|
|**影響**|權重更新過大、Loss 震盪或發散、數值溢出|
|**解決方法**|梯度裁剪、權重初始化、使用適當的激活函數、調整學習率|

---

### **總結**

梯度爆炸是深度學習中影響模型訓練的一個關鍵問題，尤其在 **深層 CNN 和 RNN** 中尤為明顯。解決梯度爆炸的方法包括 **梯度裁剪、合適的權重初始化、選擇合適的激活函數及學習率調整**，這些都是 AI 應用規劃考試可能涉及的重要知識點。

這些概念你可以用來整理教材或準備考試，如果還需要補充特定細節，歡迎詢問！