# AI-Unit Core Engine 🚀

An open-source mathematical framework and blind digital jury system designed to audit, benchmark, and optimize Large Language Model (LLM) efficiency across varied computational tasks.

---

## 🧠 Mathematical Framework

The **AI-Unit** metric evaluates models based on three core dimensions: Cognitive Difficulty ($k$), Consensus Accuracy ($A_k$), and Dynamic Speed Factor ($S_k$).

### 1. Difficulty Weight ($W_k$)
To prevent linear scaling and heavily reward deep reasoning capabilities, the task difficulty utilizes a quadratic weight:
$$W_k = k^2$$
*Where $k \in \{1, 2, 3, 4, 5\}$ represents the cognitive tier of the prompt.*

### 2. Dynamic Speed Factor ($S_k$)
Speed is measured relatively against the current market leaders' performance. The median response time ($T_{target}$) of top-tier models is checked against the tested model's actual runtime ($T_{actual}$) using a fractional bounded decay function:
$$S_k = \frac{T_{target}}{T_{actual} + T_{target}}$$
*This ensures $S_k$ is strictly bounded between $0$ and $1$.*

### 3. Total Consolidated AI-Unit Score
The final efficiency score is calculated as the sum of weighted achievements across all evaluated test cases:
$$\text{Total AI-Unit Score} = \sum (W_k \times A_k \times S_k)$$

---

## ⚖️ Blind Digital Jury Architecture

To eliminate corporate model bias, the engine decouples the evaluation process:
1. **Identity Masking:** Strips the tested model's name and metadata, assigning an anonymous random cryptographic ID.
2. **Multi-Juror Voting:** Routes the masked response to separate foundational judge models (e.g., GPT, Claude, Gemini).
3. **Absolute Majority Consensus:** Applies a strict threshold where the final accuracy score ($A_k$) is locked to $1.0$ if the majority approves, and $0.0$ if rejected.

---

## 🛠️ Core Implementation

The core engine is written in standard Python and executes inside a deterministic sandbox environment to monitor real-time resource allocations.
