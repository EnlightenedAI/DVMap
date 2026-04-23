# DVMap: Fine-Grained Pluralistic Value Alignment via High-Consensus Demographic-Value Mapping

## 📖 Overview

Current Large Language Models (LLMs) typically rely on coarse-grained national labels for pluralistic value alignment. However, such macro-level supervision often obscures **intra-country value heterogeneity**, resulting in suboptimal alignment performance.

**DVMap** is a framework designed for fine-grained pluralistic value alignment. Instead of shifting from national labels to multi-dimensional **demographic constraints**, DVMap identifies groups with **predictable, high-consensus** value preferences. This approach enables more nuanced and accurate value mapping across diverse populations.

-----

## 🚀 News

  * **[2026/04]** Core datasets and code have been uploaded and will be continuously maintained.
  * **[2026/04]** DVMap has been accepted to **ACL 2026**.

-----

## 🛠️ Quick Start

### 1\. Environment Setup

This project is built upon the **veRL** framework. Please install the dependencies using:

```bash
pip install -r DVMap/Demographic_Value_Alignment/verl/requirements.txt
```

### 2\. Training

We provide a GRPO training script based on Qwen3-4B. The script is configured with reward functions specifically tailored for demographic-value mapping:

```bash
# Start GRPO training
bash DVMap/Demographic_Value_Alignment/verl/examples/grpo_trainer/dvmap_qwen3_4b.sh
```

### 3\. Evaluation

Evaluate the trained models using the **Triple-Generalization Evaluation Benchmark**:

```bash
# Run evaluation pipeline
bash DVMap/Triple-Generalization_Evaluation_Benchmark/run_eval.sh
```

-----

## ⚖️ Implementation Details

The large-scale reinforcement learning training in this project is powered by the **[veRL](https://www.google.com/search?q=https://github.com/volcengine/verl)** (Variable Efficient RL) framework. Leveraging veRL's flexible **Hybrid Engine**, we achieved efficient GRPO training for complex value-mapping tasks with optimized generation throughput and memory efficiency on distributed clusters.

-----

## 🙏 Acknowledgements

This project is developed based on the following open-source data and frameworks:

  * **World Values Survey (WVS):** Our research deeply relies on the global value survey data provided by the WVS Association. These comprehensive empirical data points form the core of our fine-grained Demographic-Value Mapping.
  * **veRL Framework:** Our alignment algorithms are implemented on top of the veRL distributed RL framework. We appreciate the veRL team for providing excellent tools for model parallelism and RLHF engineering.

-----