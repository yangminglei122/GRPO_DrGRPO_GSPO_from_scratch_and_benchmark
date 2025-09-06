import os
import re
import copy
import random
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from datasets import load_dataset

logger = None


def setup_logging(log_dir: str):
    """
    配置日志：文件和控制台句柄
    Configure logging with file and console handlers.
    """
    # 日志文件夹
    os.makedirs(log_dir, exist_ok=True)

    global logger
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)

    # 清除所有已存在句柄
    logger.handlers = []

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    log_file = f'{log_dir}{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logging initialized")

    return logger


# 系统提示
SYSTEM_PROMPT = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"


class RLVRTrainer:
    """
    用于LLM微调的强化学习训练器RLVRTrainer
    Reinforcement Learning with Verifiable Rewards Trainer for language model fine-tuning.
    """

    def __init__(self, model_name: str, output_dir: str, seed: int = 42):
        """
        初始化训练器：模型初始化、训练参数配置
        Initialize the trainer with model and training configuration.
        """

        self._set_random_seed(seed)

        logger.info(f"初始化 trainer with model: {model_name}")
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model and tokenizer 在 setup_model() 函数中初始化
        self.model = None
        self.tokenizer = None
        self.ref_model = None

    def _set_random_seed(self, seed: int = 42):
        """
        为可复现，设置随机数种子
        Set random seeds for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_model(self):
        """
        加载和配置model和tokenizer
        Load and configure the model and tokenizer.
        """
        logger.info("下载模型...")

        # 预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="eager",  # 或 "sdpa" 关闭 FlashAttention
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )

        # Configure padding and EOS tokens
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Tokenizer pad token: {self.tokenizer.pad_token}")
        logger.info(f"Model pad token ID: {self.tokenizer.pad_token_id}")

        return self

    def optimize_memory(self):
        """
        优化训练时的模型内存
        Optimize model memory usage for training.
        """
        self.model.train()
        # 不使用cache
        self.model.config.use_cache = False
        # 开启梯度检查点
        self.model.gradient_checkpointing_enable()
        logger.info("Memory optimization applied")
        return self

    @staticmethod
    def load_dataset(split: str = "train") -> List[Dict[str, str]]:
        """
        下载和准备GSM8K数据集
        Load and prepare the GSM8K dataset.

        Args:
            split: Dataset split to load ("train" or "test")
        Returns:
            List of formatted examples with prompt and answer
        """
        logger.info(f"加载 GSM8K dataset ({split} split)...")
        # data = load_dataset("openai/gsm8k", "main")[split]
        base_path = "./gsm8k"
        data_files = {
            "train": os.path.join(base_path, "train-00000-of-00001.parquet"),
            "test": os.path.join(base_path, "test-00000-of-00001.parquet"),
        }
        dataset = load_dataset("parquet", data_files=data_files)
        data = dataset[split]

        # 格式化处理为：prompt-answer pairs
        formatted_data = []
        for example in data:
            prompt = RLVRTrainer._build_prompt(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ]
            )

            answer = RLVRTrainer._extract_answer_from_dataset(example["answer"])
            if (
                answer is not None
            ):  # Only include valid examples，只保留有明确答案的样本
                formatted_data.append({"prompt": prompt, "answer": answer})

        logger.info(f"加载样本数为： {len(formatted_data)} ")

        return formatted_data

    @staticmethod
    def _build_prompt(messages: List[Dict[str, str]]) -> str:
        """
        根据message列表，构建单个的提示词字符串
        Build a single prompt string from a list of messages.
        """
        return "\n".join([msg["content"].strip() for msg in messages])

    @staticmethod
    def _extract_answer_from_dataset(text: str) -> Optional[str]:
        """
        从GSM8K数据集中提取答案
        Extract answer from GSM8K dataset examples.
        """
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    @staticmethod
    def extract_answer_from_model_output(text: str) -> Optional[str]:
        """
        从模型输出中提取答案
        Extract the value from the last <answer> tag in the text.
        """
        parts = text.split("<answer>")
        if len(parts) < 2:
            return None

        last_part = parts[-1]
        if "</answer>" not in last_part:
            return None

        answer = last_part.split("</answer>")[0].strip()
        return None if answer == "..." else answer

    @staticmethod
    def _extract_last_number(text: str) -> Optional[float]:
        """
        从文本中提取最后一个数字
        Extract the last number appearing in the text.
        """
        text = text.replace("$", "").replace("%", "")
        pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_single_number(text: str) -> Optional[float]:
        """
        从文本中提取单个数字，如果只有一个数字
        Extract a single number from text if exactly one number is present.
        """
        numbers = re.findall(r"-?\d*\.?\d+", text)
        return float(numbers[0]) if len(numbers) == 1 else None

    def evaluate(
        self, eval_data: List[Dict[str, str]], max_new_tokens: int = 400
    ) -> float:
        """
        评估模型在一组样本上的性能
        Evaluate the model on a set of examples.

        Args:
            eval_data: List of evaluation examples
            max_new_tokens: Maximum tokens to generate

        Returns:
            Accuracy percentage
        """
        # 正确率
        correct = 0
        # 样本总数
        total = len(eval_data)
        if total == 0:
            return 0

        # 切换到评估模式
        self.model.eval()

        logger.info("\n" + "=" * 50)
        logger.info(f"在 {total} 例样本上进行评估")
        logger.info("=" * 50)

        # Evaluate each example
        for i, example in enumerate(eval_data):
            # 完整的提示
            full_prompt = example["prompt"]
            # 期望的答案
            expected = example["answer"]

            # 对完整的提示进行分词
            inputs = self.tokenizer(
                full_prompt, return_tensors="pt", padding=True, padding_side="left"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=False,
                    use_cache=True,  # TODO: remove this?
                )

            # 解码生成的文本
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取生成的答案
            predicted = self.extract_answer_from_model_output(response)

            # 检查答案是否正确
            is_correct = False
            if predicted == expected:  # 精准匹配
                is_correct = True
            else:
                # 尝试是否单个数字匹配
                pred_num = self._extract_single_number(str(predicted))
                exp_num = self._extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # 尝试最后的数字匹配
                    pred_num = self._extract_last_number(str(predicted))
                    exp_num = self._extract_last_number(str(expected))
                    is_correct = (
                        pred_num is not None
                        and exp_num is not None
                        and pred_num == exp_num
                    )

            # 更新记录
            if is_correct:
                correct += 1

            # Log evaluation details
            logger.info(
                f"\nExample {i+1}/{total} | Correct: {'✓' if is_correct else '✗'}\n"
                f"Prompt: {full_prompt}\n\n"
                f"Expected: {expected}\n\n"
                f"Completion: {response[len(full_prompt):]}\n\n"
                f"Extracted: {predicted}\n" + "-" * 50
            )

        # Calculate and print final accuracy
        accuracy = (correct / total) * 100.0
        logger.info(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
        logger.info("=" * 50)

        # 恢复训练模式
        self.model.train()
        return accuracy

    def _compute_log_probs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """
        计算模型输出的特定词汇的对数概率
        Compute log probabilities for specific tokens in the vocabulary.

        Args:
            model: Language model
            input_ids: Token IDs for input sequences
            attention_mask: Attention mask for input sequences
            logits_to_keep: Number of tokens to keep from the end

        Returns:
            Log probabilities of the selected tokens
        """
        # Get logits for all tokens except the last one.
        #
        # DETAILED EXPLANATION:
        # In an autoregressive causal language model, the goal is to predict the *next*
        # token in a sequence given the preceding tokens. When we feed a sequence of
        # tokens into the model, it outputs a sequence of logits of the same length.
        # The logit at position `i` is the model's prediction for the token at position `i+1`.
        #
        # Let's visualize this with a simple example sequence of token IDs:
        #
        # Input IDs:  [t_0, t_1, t_2, t_3]  (length 4)
        #
        # The model processes this and produces a logit for each input token position:
        #
        # Logits:     [L_0, L_1, L_2, L_3]  (length 4)
        #
        # Here's what each logit represents as a prediction:
        #
        # - L_0 is generated after seeing t_0. It's the prediction for the *next* token, which should be t_1.
        # - L_1 is generated after seeing t_0, t_1. It's the prediction for the *next* token, which should be t_2.
        # - L_2 is generated after seeing t_0, t_1, t_2. It's the prediction for the *next* token, which should be t_3.
        # - L_3 is generated after seeing t_0, t_1, t_2, t_3. It's the prediction for the token that would come *after* t_3.
        #
        # When calculating the loss, we compare the model's predictions (logits) against the
        # actual "next tokens" (the labels). The labels are simply the input sequence shifted
        # one position to the left.
        #
        #   Predictions:      L_0      L_1      L_2      L_3
        #                      |        |        |        |
        #   Target Labels:    t_1      t_2      t_3     ???
        #
        # As the diagram shows, the last logit (L_3) has no corresponding target label in our
        # original sequence. There is no "t_4" to compare it against. Therefore, this last
        # logit is irrelevant for calculating the loss of the current sequence.
        #
        # To align the predictions with the labels for loss computation, we must:
        # 1. Discard the last logit from the predictions.
        # 2. Discard the first token from the labels (as it was never predicted).
        #
        # In PyTorch slicing:
        # - `logits[:, :-1, :]` keeps all logits except for the last one on the sequence dimension.
        # - `labels[:, 1:]` keeps all labels except for the first one on the sequence dimension.
        #
        # This is why the code slices the logits tensor to exclude the final token's logit.

        # 模型推理得到的logits，舍弃最后一个
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
            :, :-1, :
        ]

        # 只选择我们所需的token的logits
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]

        # 计算log softmax并获取相关概率
        # log_probs的shape是[batch_size, logits_to_keep, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # 返回的shape是[batch_size, logits_to_keep]
        return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    def _create_completion_mask(
        self, completion_ids: torch.Tensor, eos_token_id: int
    ) -> torch.Tensor:
        """
        构建mask:摒弃EOS token之后的部分
        Create a mask for completion tokens that excludes tokens after the EOS token.

        Args:
            completion_ids: Token IDs of the generated completions
            eos_token_id: ID of the end-of-sequence token

        Returns:
            二值mask：1表示有效token，0表示EOS之后的token
            Binary mask with 1s for valid tokens and 0s after EOS
        """
        # Check if there is an EOS token in each sequence
        is_eos = completion_ids == eos_token_id
        batch_size = is_eos.size(0)

        # 每个sequence里寻找第一个EOS token的index
        eos_idx = torch.full(
            (batch_size,),
            completion_ids.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        # mask_exists 的shape是[batch_size]，表示是否有EOS token
        mask_exists = is_eos.any(dim=1)
        # 将 eos_idx 中对应位置的值更新为 is_eos 中每个样本第一个 True 出现的索引位置
        if mask_exists.any():
            eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

        # Create mask: 1 for tokens before and including EOS, 0 otherwise
        sequence_indices = torch.arange(
            completion_ids.size(1), device=completion_ids.device
        ).expand(batch_size, -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def _generate_completions(
        self, prompts: List[str], num_generations: int, max_completion_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成多个提示词的补全
        Generate multiple completions for each prompt.

        Args:
            prompts: List of text prompts
            num_generations: Number of completions per prompt
            max_completion_length: Maximum tokens to generate

        Returns:
            Tuple of (prompt_ids, prompt_mask, completion_ids, completion_mask)
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.device)

        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)

        # 根据num_generations，重复prompt_ids和prompt_mask
        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

        # 生成补全时，禁用梯度检查点，以提高效率
        self.model.gradient_checkpointing_disable()
        # We don't need “with torch.no_grad()” here,
        # because in the upper layer calling function _generate_rollout_data,
        # there is already a “with torch.no_grad()” block.

        # 此处不需要“with torch.no_grad()”,因为：
        # 1. 这里是在_generate_rollout_data函数中调用的，_generate_rollout_data 函数中已经有了“with torch.no_grad()”块
        # 2. 这里的模型推理已经在 _generate_rollout_data 函数中完成，所以这里不需要再次进行模型推理
        outputs = self.model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=False,
            use_cache=True,
        )
        self.model.gradient_checkpointing_enable()

        # Extract completions (excluding prompt tokens)
        # 提取补全序列，去掉提示词的部分
        completion_ids = outputs[:, prompt_length:]
        completion_mask = self._create_completion_mask(
            completion_ids, self.tokenizer.eos_token_id
        )

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate_rollout_data(
        self,
        batch_samples: List[Dict[str, str]],
        num_generations: int,
        max_completion_length: int,
    ) -> Dict[str, Any]:
        """
        为rollouts生成数据：包括补全和对数概率
        Generate data for rollouts including completions and log probabilities.

        Args:
            batch_samples: Batch of training samples
            num_generations: Number of completions per sample
            max_completion_length: Maximum completion length

        Returns:
            Dictionary containing all data needed for updates
        """
        prompts = [sample["prompt"] for sample in batch_samples]
        answers = [sample["answer"] for sample in batch_samples]

        with torch.no_grad():
            # 生成补全，即响应
            prompt_ids, prompt_mask, completion_ids, completion_mask = (
                self._generate_completions(
                    prompts, num_generations, max_completion_length
                )
            )

            # 将prompt和completion合并成一个序列
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            # 计算log prob：策略model和ref model
            logits_to_keep = completion_ids.size(1)
            old_log_probs = self._compute_log_probs(
                self.model, input_ids, attention_mask, logits_to_keep
            )
            ref_log_probs = self._compute_log_probs(
                self.ref_model, input_ids, attention_mask, logits_to_keep
            )

        # 为计算reward，格式化completions，不需要提示词的部分
        formatted_completions = [
            [{"content": self.tokenizer.decode(ids, skip_special_tokens=True)}]
            for ids in completion_ids
        ]

        # 重复prompts和answers，以匹配生成的number of generations
        repeated_prompts = [p for p in prompts for _ in range(num_generations)]
        repeated_answers = [a for a in answers for _ in range(num_generations)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "formatted_completions": formatted_completions,
            "repeated_prompts": repeated_prompts,
            "repeated_answers": repeated_answers,
            "logits_to_keep": logits_to_keep,
            "batch_size": len(prompts),
            "num_generations": num_generations,
        }

    def _correctness_reward(
        self,
        prompts: List[str],
        completions: List[List[Dict[str, str]]],
        answers: List[str],
    ) -> List[float]:
        """
        正确率奖励
        计算奖励：根据提取的答案的正确性。
        拉齐准确率奖励：精准匹配高奖励，数字匹配中奖励，其他奖励为0。
        """
        responses = [completion[0]["content"] for completion in completions]
        extracted = [self.extract_answer_from_model_output(r) for r in responses]

        rewards = []
        for r, a in zip(extracted, answers):
            if r == a:  # Exact match
                rewards.append(2.0)
            else:
                # Try numeric equivalence
                r_num = self._extract_single_number(str(r))
                a_num = self._extract_single_number(str(a))
                if r_num is not None and a_num is not None and r_num == a_num:
                    rewards.append(1.5)
                else:
                    rewards.append(0.0)
        return rewards

    def _format_reward(self, completions: List[List[Dict[str, str]]]) -> List[float]:
        """
        格式奖励：为遵循所需的 XML 格式分配奖励。
        """
        responses = [completion[0]["content"] for completion in completions]

        rewards = []
        for response in responses:
            score = 0.0
            if "<reasoning>" in response:
                score += 0.2
            if "</reasoning>" in response:
                score += 0.2
            if "<answer>" in response:
                score += 0.2
            if "</answer>" in response:
                score += 0.2
            rewards.append(score)
        return rewards

    def _combined_reward(
        self,
        prompts: List[str],
        completions: List[List[Dict[str, str]]],
        answers: List[str],
    ) -> List[float]:
        """
        合并奖励：正确率奖励和格式奖励的加权和。
        """
        correctness_scores = self._correctness_reward(prompts, completions, answers)
        format_scores = self._format_reward(completions)

        return [c + f for c, f in zip(correctness_scores, format_scores)]

    def _compute_loss(
        self,
        rollout_data: Dict[str, Any],
        beta: float = 0.01,
        epsilon: float = 0.2,
        loss_type: str = "grpo",
        importance_sampling_level: str = "token",
    ) -> Tuple[torch.Tensor, float]:
        """
        计算损失：用于策略梯度更新

        Args:
            rollout_data: _generate_rollout_data()生成的数据
            beta: KL penalty 系数
            epsilon: Clipping parameter for PPO
            loss_type: Type of loss to use ("grpo", "dr_grpo", or "gspo")
            importance_sampling_level: Level of importance sampling ("token" or "sequence")

        Returns:
            Tuple of (loss tensor, average reward)
        """

        # sequence-level importance weighting只用于GSPO
        if importance_sampling_level == "sequence":
            assert loss_type == "gspo"

        # 计算当前token的log概率
        token_log_probs = self._compute_log_probs(
            self.model,
            rollout_data[
                "input_ids"
            ],  # shape is [batch_size * num_generations, max_len_of_input_prompts + max_completion_length], e.g. [16, 487]
            rollout_data[
                "attention_mask"
            ],  # shape is [batch_size * num_generations, max_len_of_input_prompts + max_completion_length], e.g. [16, 487]
            rollout_data["logits_to_keep"],  # max_completion_length, e.g. 400
        )
        # token_log_probs.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]

        # 计算概率比率，即所谓重要性
        if importance_sampling_level == "token":
            # 先log在exp，得到概率比率
            ratio = torch.exp(token_log_probs - rollout_data["old_log_probs"])
        else:
            # Calculate the sequence-level log ratio per paper's formula (7).
            # GSPO's objective is sequence-level. However, by broadcasting the single
            # sequence-level ratio (seq_ratio) to every token in the sequence,
            # we can reuse the token-level PPO loss computation framework.
            # The resulting gradient is mathematically equivalent to a sequence-level objective
            # because all tokens within a sequence share the same advantage and the same ratio.

            # 计算序列级的 log ratio
            log_ratio = (
                token_log_probs - rollout_data["old_log_probs"]
            )  # log_ratio.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]

            # 1. 使用 completion_mask 屏蔽 padding tokens
            # 2. 沿序列维度求和，然后除以序列的实际长度
            # 此计算方式参考原文
            sum_log_ratio = (log_ratio * rollout_data["completion_mask"]).sum(
                dim=1
            )  # sum_log_ratio.shape is [batch_size * num_generations], e.g. [16]
            seq_len = (
                rollout_data["completion_mask"].sum(dim=1).clamp(min=1.0)
            )  # clamp 避免除零. seq_len.shape is [batch_size * num_generations], e.g. [16]
            seq_ratio = torch.exp(
                sum_log_ratio / seq_len
            )  # seq_ratio.shape is [batch_size * num_generations], e.g. [16]

            # 将序列级的 ratio 广播到每个 token 上
            ratio = seq_ratio.unsqueeze(-1).expand_as(
                token_log_probs
            )  # ratio.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]

        # 计算奖励
        rewards = torch.tensor(
            self._combined_reward(
                rollout_data["repeated_prompts"],
                rollout_data["formatted_completions"],
                rollout_data["repeated_answers"],
            ),
            dtype=torch.float32,
            device=self.device,
        )  # rewards.shape is [batch_size * num_generations], e.g. [16]

        # 调整奖励的shape，使其与token_log_probs的shape相同
        # 计算优势
        batch_size = rollout_data["batch_size"]  # batch_size is 2
        num_generations = rollout_data["num_generations"]  # num_generations is 8
        rewards = rewards.view(batch_size, num_generations)  # rewards.shape is [2, 8]

        # 计算平均奖励
        avg_reward = rewards.mean().item()  # avg_reward is a float
        logger.info(f"Average Reward: {avg_reward:.6f}")

        # 计算优势
        if loss_type == "grpo" or loss_type == "gspo":
            # Original GRPO: standardize rewards within each prompt
            # GSPO: https://arxiv.org/pdf/2507.18071
            mean_rewards = rewards.mean(dim=1).repeat_interleave(
                num_generations
            )  # mean_rewards.shape = [batch_size * num_generations], e.g. [16]
            std_rewards = rewards.std(dim=1, unbiased=False).repeat_interleave(
                num_generations
            )  # std_rewards.shape = [batch_size * num_generations], e.g. [16]
            std_rewards = torch.clamp(std_rewards, min=1e-4)  # Avoid division by zero
            advantages = ((rewards.view(-1) - mean_rewards) / std_rewards).unsqueeze(
                1
            )  # advantages.shape = [batch_size * num_generations, 1], e.g. [16, 1]
        elif loss_type == "dr_grpo":
            # DrGRPO: remove standardization, use reward difference directly
            mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
            advantages = (rewards.view(-1) - mean_rewards).unsqueeze(1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # 使用裁剪（clip）计算近端策略优化（PPO）替代目标。
        surr1 = (
            ratio * advantages
        )  # surr1.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]
        surr2 = (
            torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        )  # surr2.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]
        surrogate_loss = torch.min(surr1, surr2)

        # 逐token计算KL散度（近似：exp(δ)−δ−1）
        kl = (
            torch.exp(rollout_data["ref_log_probs"] - token_log_probs)
            - (rollout_data["ref_log_probs"] - token_log_probs)
            - 1
        )  # rollout_data["ref_log_probs"].shape and token_log_probs.shape are [batch_size * num_generations, max_completion_length], e.g. [16, 400]
        # kl.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]

        # 组合完整loss
        per_token_loss = (
            surrogate_loss - beta * kl
        )  # per_token_loss.shape is [batch_size * num_generations, max_completion_length], e.g. [16, 400]
        loss = -(
            (per_token_loss * rollout_data["completion_mask"]).sum(dim=1)
            / rollout_data["completion_mask"].sum(dim=1)
        ).mean()  # (per_token_loss * rollout_data["completion_mask"]).sum(dim=1).shape is [batch_size * num_generations], e.g. [16]
        # loss.shape is []

        return loss, avg_reward

    def train(
        self,
        train_data: List[Dict[str, str]],
        num_iterations: int,
        num_steps: int,
        batch_size: int,
        num_generations: int,
        max_completion_length: int,
        beta: float,
        learning_rate: float,
        epsilon: float,
        loss_type: str,  # Can be "grpo", "dr_grpo", or "gspo"
        importance_sampling_level: str,  # Can be "token" or "sequence"
    ):
        """
        使用RLVR算法训练模型(e.g. GRPO, DrGRPO, GSPO).

        Args:
            train_data: 训练数据集
            num_iterations: 外循环次数 (reference model updates)
            num_steps: 每个iteration中batch更新次数，内循环次数
            batch_size: Number of prompts per batch
            num_generations: Number of completions per prompt
            max_completion_length: Maximum token length for completions
            beta: KL penalty coefficient
            learning_rate: Learning rate for optimizer
            epsilon: PPO clipping parameter
            loss_type: Type of loss to use ("grpo", "dr_grpo", or "gspo")
            importance_sampling_level: Level of importance sampling ("token" or "sequence")
        """
        logger.info("\n 开始RL训练 ...")
        logger.info(
            f"使用算法: {loss_type.upper()} + {importance_sampling_level}-水平重要性采样"
        )

        for iteration in range(num_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"ITERATION {iteration+1}/{num_iterations}")
            logger.info(f"{'='*50}")

            # 构建参考模型：每次外循环，更新参考模型
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("已构建参考模型")

            # optimizer初始化
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            self.model.train()

            # 训练内循环
            for step in range(num_steps):
                # 采样一个batch
                batch_samples = random.sample(train_data, batch_size)

                # 生成 rollout data
                rollout_data = self._generate_rollout_data(
                    batch_samples, num_generations, max_completion_length
                )

                # 策略更新
                loss, avg_reward = self._compute_loss(
                    rollout_data,
                    beta=beta,
                    epsilon=epsilon,
                    loss_type=loss_type,
                    importance_sampling_level=importance_sampling_level,
                )

                # 梯度回传
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                optimizer.step()

                # 记录训练日志
                logger.info(
                    f"Iter: {iteration+1}/{num_iterations}, "
                    f"Step: {step+1}/{num_steps}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Avg Reward: {avg_reward:.6f}, "
                    f"Algorithm: {loss_type.upper()}"
                )

        # 模型保存
        logger.info("\n保存微调模型...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"保存微调模型到文件夹：{self.output_dir}")

    def test_model(self, prompts: List[str], max_new_tokens: int = 400):
        """
        测试微调模型，基于example prompts
        """
        self.model.eval()
        logger.info("\n" + "=" * 50)
        logger.info("模型测试")
        logger.info("=" * 50)

        for i, prompt in enumerate(prompts):
            # Prepare the prompt
            test_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            test_prompt = self._build_prompt(test_messages)

            # Generate response
            inputs = self.tokenizer(
                test_prompt, return_tensors="pt", padding=True, padding_side="left"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    early_stopping=False,
                    use_cache=True,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            extracted_answer = self.extract_answer_from_model_output(response)

            # Log results

            logger.info(
                f"Test #{i+1} | Prompt: {prompt}\n\n"
                f"Full completion: {response[len(test_prompt):]}\n\n"
                f"Extracted: {extracted_answer}\n" + "-" * 50
            )

        self.model.train()
