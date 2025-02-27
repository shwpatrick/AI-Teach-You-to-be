## **ReLU 在深度學習中的應用**

- **CNN（卷積神經網絡）**：ReLU 是 CNN 最常用的激活函數，能夠加速訓練並提高模型效果。
- **RNN（循環神經網絡）**：ReLU 由於梯度爆炸問題較嚴重，不適合 RNN，通常用 **tanh** 或 **LSTM** 變種。
- **Transformer**：BERT、GPT 等模型中，部分結構（如 Feedforward Network）使用 GELU（類似於 ReLU）。

---

## **7. ReLU 的考點**

### **(1) ReLU 的數學公式**

- 需要記住 **ReLU 及其導數** 的公式： f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x) f′(x)={1,x>00,x≤0f'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}f′(x)={1,0,​x>0x≤0​

### **(2) ReLU 的優勢**

- 為何 ReLU 比 sigmoid 和 tanh 更受歡迎？
- 如何解釋 ReLU 的 **梯度消失問題較少**，但仍有 **死亡神經元問題**？

### **(3) 死亡神經元問題**

- 什麼是 **Dying ReLU**？
- 如何用 **Leaky ReLU、PReLU、ELU** 來解決？

### **(4) ReLU 的變體**

- Leaky ReLU 與 PReLU 的區別？
- ELU 為何能減少梯度為 0 的問題？
- GELU 為何在 Transformer 中常用？

### **(5) ReLU 在深度學習中的應用**

- 為什麼 CNN 中常用 ReLU？
- RNN 中為何不適用 ReLU？
- Transformer 為何選擇 GELU 而非 ReLU？

---

## **8. 小結**

- **ReLU 的核心公式**：f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x)
- **主要優勢**：
    - 計算簡單，避免梯度消失問題。
    - 在 CNN 等深度學習模型中表現優異。
- **主要缺點**：
    - **死亡神經元問題**：解法包括 Leaky ReLU、PReLU、ELU。
    - **輸出均值偏移問題**：可透過 Batch Normalization 解決。
- **考點**：數學推導、死亡神經元問題、變體比較、適用場景。

ReLU 是深度學習中最重要的激活函數之一，熟練掌握它的特性，能夠幫助更好地理解深度學習模型的訓練機制！