# 🧠 LLM-based Security Patch Detection

This repository contains the implementation and datasets used in the paper.  
It provides all necessary components for reproducing the experimental results, including dataset organization, agent implementation, model configuration, and method setup.

---

## 📂 Dataset

All dataset files are stored in the **`Dataset/`** directory.  

**Naming Convention:**
- The numeric prefix in each filename corresponds to the associated **CWE (Common Weakness Enumeration)** category.  
- **Example:**  
20_patch_db.json → CWE-20
  
## 🤖 Agent Implementation

The implementation of the agent-based framework is provided in the **`agent/`** directory.  
This module manages task coordination, model interaction, and iterative reasoning across different LLM configurations.

---

## ⚙️ Model Configuration
### ☁️ Cloud-based Models
- Implementations based on **DeepSeek** and **GPT** require updating the corresponding **API keys** in the configuration files prior to execution.

### 💻 Local Models
- Implementations based on **Llama-3** and **Gemma-3** require specifying the **local model paths** to ensure proper loading and inference.

---

## 🧩 LLM Method Implementation

Implementations of both **Plain LLM** and **Data-Aug LLM** methods are located in the **`LLM/`** directory.  
These modules define the experimental setups used to evaluate the effectiveness of LLMs under different prompting and data augmentation strategies.

---

## 🔧 Configuration Details

### Model Setup
- **DeepSeek / GPT:** Update the corresponding **API keys** in the configuration files.  
- **Llama-3 / Gemma-3:** Specify the **local model directories** for inference.  

### Method Configuration
- Each method may require minor adjustments to **prompt templates** depending on experimental objectives.  
- The framework supports both **Plain LLM** and **Data-Aug LLM** configurations, enabling flexible experimentation and ablation analysis.

---
