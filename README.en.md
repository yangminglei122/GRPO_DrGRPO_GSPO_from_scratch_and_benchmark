## Write Some Code | Comparing GRPO, DrGRPO, and GSPO from Scratch on the Same Dataset and Training Steps  
  
**[本文档中文版](README.md)**   

  
In the field of Reinforcement Learning (RL) for Large Language Models (LLMs), an exciting direction is model optimization using Verifiable Rewards, known as RLVR. Traditional Reinforcement Learning from Human Feedback (RLHF) or a dedicated Value Model can be subjective and expensive. In contrast, RLVR guides model learning through programmatic, objective reward functions. For instance, in mathematics, the correctness of an answer can be verified by computation. This approach offers a more efficient and scalable path to enhance complex abilities like reasoning in models.

Guided by the core idea of RLVR, a series of excellent algorithms have emerged, with GRPO, DrGRPO, and GSPO being the most representative. GRPO is the core algorithm for training DeepSeek V2, while GSPO is the core algorithm for training Qwen2. They both evolve from the classic PPO algorithm but have taken different exploratory paths in pursuit of higher efficiency and stability. 

![alt text](images/grpo_drgrpo_gspo.png)

```bash
%git clone https://github.com/zhangfaen/GRPO_DrGRPO_GSPO_from_scratch_and_benchmark
%cd GRPO_DrGRPO_GSPO_from_scratch_and_benchmark
%conda create -n grpo_drgrpo_gspo python=3.12
%conda activate grpo_drgrpo_gspo 
%pip install -r requirements.txt
%python GRPO_DrGRPO_GSPO_from_scratch_and_benchmark.py
```

### GRPO, DrGRPO, and GSPO: A Shared Lineage, Each with Its Own Merits

To understand these three, we must first understand their common ancestor: **GRPO (Group Relative Policy Optimization)**.

**The core idea of GRPO** is to discard the value model that requires separate training in PPO, thereby significantly reducing computational and memory overhead. Its approach is quite ingenious: for the same prompt, the model generates a group of answers, which are then scored by a reward function. Instead of predicting an absolute "value," it calculates the "advantage" of each answer relative to the average score of the group. If an answer's score is above the average, it gains a positive advantage, and the model is encouraged to learn a policy that generates similar answers; the opposite is true for scores below the average. This concept of "intra-group relative comparison" is the origin of the name GRPO, and it makes the training process more stable and efficient.

However, some researchers believe that GRPO's original design has inherent biases. Its loss function calculation systematically "favors" longer responses among incorrect answers and gives excessive weight to problems that are either too easy or too hard (i.e., where all generated answers are either correct or incorrect).

To address these issues, **DrGRPO (GRPO Done Right)** was developed. Its improvement is straightforward: **remove the operations that cause bias**. Specifically, DrGRPO removes the normalization by standard deviation in the advantage calculation and the normalization by sequence length in the loss function, thereby achieving a fairer and more unbiased optimization objective.

While GRPO and DrGRPO are efficient, some researchers argue they share a deeper problem: **a mismatch between the granularity of rewards and optimization**. The reward is given for the entire generated sequence (e.g., whether the final answer is correct), but the optimization is performed at the token level. This mismatch can introduce significant noise in high-difficulty tasks and more complex models (like Mixture-of-Experts, or MoE), leading to highly unstable training and even model collapse.

This led to the emergence of **GSPO (Group Sequence Policy Optimization)**, which aims to solve this problem at its root. **The core of GSPO is to elevate the optimization granularity from the token level to the sequence level**. It no longer calculates an importance weight for each token but computes a single, unified weight for the entire sequence. This aligns the optimization objective perfectly with the reward mechanism. All update operations, including the clipping in PPO, are performed at the sequence level. This change greatly enhances training stability, especially for MoE models, avoiding complex techniques required by GRPO like "Routing Replay" and significantly improving training efficiency and final performance.

### Code Introduction: GRPO_DrGRPO_GSPO_from_scratch_and_benchmark.py

To help everyone gain a deeper understanding and feel for the differences between these three algorithms, I have written a Python script named `GRPO_DrGRPO_GSPO_from_scratch_and_benchmark.py`. This file aims to provide a clear, runnable environment for you to practice and compare these cutting-edge RL algorithms firsthand. **Note: The primary purpose of this code is for learning and understanding. Please modify and optimize it for practical use.**

The script mainly consists of the following parts:

1.  **A Unified Trainer `RLVRTrainer`**: For a fair comparison, I have encapsulated the common training workflow of the three algorithms into a single `RLVRTrainer` class. It covers all necessary steps, including loading the model and tokenizer, data processing, logging, model evaluation, and saving the final model.

2.  **Clear Algorithm Implementations**: In the `_compute_loss` method, you can clearly see the core differences in the loss function calculations of the three algorithms by using the `loss_type` parameter (options: "grpo", "dr_grpo", "gspo"). For GSPO, the `importance_sampling_level` parameter is used to distinguish its sequence-level importance sampling implementation. The code logic is designed to be consistent with the original ideas of the algorithms, making it easy to understand alongside the papers.

3.  **Standardized Experimental Setup**: The script uses the public `openai/gsm8k` dataset for training and evaluation on mathematical reasoning tasks. In the `main` function, you will see a standardized experimental process:
    *   **Load and Prepare Data**: The dataset is loaded once and split into training and evaluation sets.
    *   **Unified Starting Point**: All algorithms start training from the same pretrained model (`Qwen/Qwen2.5-1.5B-Instruct`) and undergo an initial performance evaluation to ensure a fair starting point for comparison.
    *   **Identical Training Resources**: All algorithms are trained using the same hyperparameters (e.g., learning rate, batch size) and for the same number of steps (`num_steps`).
    *   **End-to-End Comparison**: The script automatically runs the training and evaluation processes for GSPO, GRPO, and DrGRPO in sequence and prints a clear comparison of the performance results at the end, including initial accuracy, final accuracy, and the improvement margin.

By reading and running this script, you can not only deepen your understanding of the core ideas behind GRPO, DrGRPO, and GSPO but also directly observe the significant advantages of "sequence-level optimization" over "token-level optimization." I hope this code helps you better understand reinforcement learning for large models.

### My Results from Running the Above Script
Running the script on an A800 GPU card took about 5 hours and produced the following results.
<div align="center">

| Method  | Initial Accuracy | Final Accuracy | Improvement (Δ) |
|---------|------------------|----------------|-----------------|
| GSPO    | 12.00%           | 72.00%         | 60.00%          |
| GRPO    | 12.00%           | 72.00%         | 60.00%          |
| DrGRPO  | 12.00%           | 58.00%         | 46.00%          |

</div>

As you can see, without reinforcement training, Qwen/Qwen2.5-1.5B-Instruct has an accuracy of about 12.00% on the mathematical reasoning task (evaluated on the openai/gsm8k dataset). After reinforcement training with the GSPO and GRPO algorithms, the accuracy increased to 72.00% for both. With DrGRPO, the accuracy improved to 58.00%. The improvement for GSPO and GRPO was 60.00%, while for DrGRPO it was 46.00%. It is worth noting that this run used the `openai/gsm8k` dataset, which consists of elementary school-level math word problems. The dataset is small, the reasoning difficulty is low, and the training was only for 200 steps. The results should be taken as a reference and do not imply that DrGRPO is inferior to GRPO and GSPO in large-scale production environments.

### Appendix
---
- [GSPO Paper](https://arxiv.org/abs/2507.18071)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Dr.GRPO Paper](https://arxiv.org/pdf/2503.20783)