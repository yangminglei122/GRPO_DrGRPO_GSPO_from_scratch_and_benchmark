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
    """Configure logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)

    global logger
    logger = logging.getLogger('Trainer')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = f'{log_dir}{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logging initialized")

SYSTEM_PROMPT = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"

class RLVRTrainer:
    """Reinforcement Learning with Verifiable Rewards Trainer for language model fine-tuning."""
        
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        seed: int = 42
    ):
        """Initialize the trainer with model and training configuration."""
        
        self._set_random_seed(seed)
        
        logger.info(f"Initializing trainer with model: {model_name}")
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model and tokenizer will be initialized in setup_model()
        self.model = None
        self.tokenizer = None
        self.ref_model = None
    

    def _set_random_seed(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_model(self):
        """Load and configure the model and tokenizer."""
        logger.info("Downloading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side="left"
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
        """Optimize model memory usage for training."""
        self.model.train()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        logger.info("Memory optimization applied")
        return self

    @staticmethod
    def load_dataset(split: str = "train") -> List[Dict[str, str]]:
        """
        Load and prepare the GSM8K dataset.
        
        Args:
            split: Dataset split to load ("train" or "test")
            
        Returns:
            List of formatted examples with prompt and answer
        """
        logger.info(f"Loading GSM8K dataset ({split} split)...")
        data = load_dataset('openai/gsm8k', 'main')[split]
        
        formatted_data = []
        for example in data:
            prompt = RLVRTrainer._build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ])
            
            answer = RLVRTrainer._extract_answer_from_dataset(example["answer"])
            if answer is not None:  # Only include valid examples
                formatted_data.append({
                    "prompt": prompt,
                    "answer": answer
                })
        
        logger.info(f"Loaded {len(formatted_data)} examples")
        return formatted_data

    @staticmethod
    def _build_prompt(messages: List[Dict[str, str]]) -> str:
        """Build a single prompt string from a list of messages."""
        return "\n".join([msg["content"].strip() for msg in messages])

    @staticmethod
    def _extract_answer_from_dataset(text: str) -> Optional[str]:
        """Extract answer from GSM8K dataset examples."""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    @staticmethod
    def extract_answer_from_model_output(text: str) -> Optional[str]:
        """Extract the value from the last <answer> tag in the text."""
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
        """Extract the last number appearing in the text."""
        text = text.replace('$', '').replace('%', '')
        pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_single_number(text: str) -> Optional[float]:
        """Extract a single number from text if exactly one number is present."""
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return float(numbers[0]) if len(numbers) == 1 else None

    def evaluate(self, eval_data: List[Dict[str, str]], max_new_tokens: int = 400) -> float:
        """
        Evaluate the model on a set of examples.
        
        Args:
            eval_data: List of evaluation examples
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Accuracy percentage
        """
        self.model.eval()
        correct = 0
        total = len(eval_data)
        
        logger.info("\n" + "="*50)
        logger.info(f"EVALUATION ON {total} EXAMPLES")
        logger.info("="*50)
        
        for i, example in enumerate(eval_data):
            full_prompt = example["prompt"]
            expected = example["answer"]
            
            # Tokenize and generate response
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left"
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
                    use_cache=True # TODO: remove this?
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = self.extract_answer_from_model_output(response)
            
            # Check correctness
            is_correct = False
            if predicted == expected:  # Exact match
                is_correct = True
            else:
                # Try single number matching
                pred_num = self._extract_single_number(str(predicted))
                exp_num = self._extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number matching
                    pred_num = self._extract_last_number(str(predicted))
                    exp_num = self._extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and
                                  pred_num == exp_num)
            
            # Update counter
            if is_correct:
                correct += 1
                
            # Log evaluation details
            
            logger.info(
                f"\nExample {i+1}/{total} | Correct: {'✓' if is_correct else '✗'}\n"
                f"Prompt: {full_prompt}\n\n"
                f"Expected: {expected}\n\n"
                f"Completion: {response[len(full_prompt):]}\n\n"
                f"Extracted: {predicted}\n"
                + "-"*50
            )
        
        # Calculate and print final accuracy
        accuracy = (correct / total) * 100
        logger.info(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
        logger.info("="*50)
        
        self.model.train()
        return accuracy

    def _compute_log_probs(
        self, 
        model: PreTrainedModel, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        logits_to_keep: int
    ) -> torch.Tensor:
        """
        Compute log probabilities for specific tokens in the vocabulary.
        
        Args:
            model: Language model
            input_ids: Token IDs for input sequences
            attention_mask: Attention mask for input sequences
            logits_to_keep: Number of tokens to keep from the end
            
        Returns:
            Log probabilities of the selected tokens
        """
        # Get logits for all tokens except the last one
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        
        # Select only the tokens we need
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        
        # Compute log softmax and gather the relevant probabilities
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    def _create_completion_mask(
        self, 
        completion_ids: torch.Tensor, 
        eos_token_id: int
    ) -> torch.Tensor:
        """
        Create a mask for completion tokens that excludes tokens after the EOS token.
        
        Args:
            completion_ids: Token IDs of the generated completions
            eos_token_id: ID of the end-of-sequence token
            
        Returns:
            Binary mask with 1s for valid tokens and 0s after EOS
        """
        is_eos = completion_ids == eos_token_id
        batch_size = is_eos.size(0)
        
        # Find the index of the first EOS token in each sequence
        eos_idx = torch.full((batch_size,), completion_ids.size(1), dtype=torch.long, device=completion_ids.device)
        mask_exists = is_eos.any(dim=1)
        if mask_exists.any():
            eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
        
        # Create mask: 1 for tokens before and including EOS, 0 otherwise
        sequence_indices = torch.arange(completion_ids.size(1), device=completion_ids.device).expand(batch_size, -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def _generate_completions(
        self,
        prompts: List[str],
        num_generations: int,
        max_completion_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple completions for each prompt.
        
        Args:
            prompts: List of text prompts
            num_generations: Number of completions per prompt
            max_completion_length: Maximum tokens to generate
            
        Returns:
            Tuple of (prompt_ids, prompt_mask, completion_ids, completion_mask)
        """
        # Encode prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left"
        ).to(self.device)
        
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        
        # Repeat prompts for multiple generations
        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
        
        # Generate completions
        self.model.gradient_checkpointing_disable()
        outputs = self.model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=False,
            use_cache=True
        )
        self.model.gradient_checkpointing_enable()
        
        # Extract completions (excluding prompt tokens)
        completion_ids = outputs[:, prompt_length:]
        completion_mask = self._create_completion_mask(
            completion_ids, 
            self.tokenizer.eos_token_id
        )
        
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate_rollout_data(
        self,
        batch_samples: List[Dict[str, str]],
        num_generations: int,
        max_completion_length: int
    ) -> Dict[str, Any]:
        """
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
            # Generate completions
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_completions(
                prompts, num_generations, max_completion_length
            )
            
            # Combine prompt and completion tokens
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            
            # Compute log probabilities
            logits_to_keep = completion_ids.size(1)
            old_log_probs = self._compute_log_probs(
                self.model, input_ids, attention_mask, logits_to_keep
            )
            ref_log_probs = self._compute_log_probs(
                self.ref_model, input_ids, attention_mask, logits_to_keep
            )
        
        # Format completions for reward calculation
        formatted_completions = [
            [{'content': self.tokenizer.decode(ids, skip_special_tokens=True)}] 
            for ids in completion_ids
        ]
        
        # Repeat prompts and answers to match number of generations
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
            "num_generations": num_generations
        }

    def _correctness_reward(
        self, 
        prompts: List[str], 
        completions: List[List[Dict[str, str]]], 
        answers: List[str]
    ) -> List[float]:
        """
        Calculates rewards based on the correctness of the extracted answer.
        Assigns a high reward for exact matches, a medium reward for numerically equivalent matches, and zero otherwise.
        """
        responses = [completion[0]['content'] for completion in completions]
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
        """Assign reward for adhering to the desired XML format."""
        responses = [completion[0]['content'] for completion in completions]
        
        rewards = []
        for response in responses:
            score = 0.0
            if "<reasoning>" in response: score += 0.2
            if "</reasoning>" in response: score += 0.2
            if "<answer>" in response: score += 0.2
            if "</answer>" in response: score += 0.2
            rewards.append(score)
        return rewards

    def _combined_reward(
        self, 
        prompts: List[str], 
        completions: List[List[Dict[str, str]]], 
        answers: List[str]
    ) -> List[float]:
        """Combine correctness and format rewards."""
        correctness_scores = self._correctness_reward(prompts, completions, answers)
        format_scores = self._format_reward(completions)
        
        return [c + f for c, f in zip(correctness_scores, format_scores)]

    def _compute_loss(
        self,
        rollout_data: Dict[str, Any],
        beta: float = 0.01,
        epsilon: float = 0.2,
        loss_type: str = "grpo",
        importance_sampling_level: str = "token"
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the loss for updating the policy model.
        
        Args:
            rollout_data: Data generated by _generate_rollout_data
            beta: KL penalty coefficient
            epsilon: Clipping parameter for PPO
            loss_type: Type of loss to use ("grpo", "dr_grpo", or "gspo")
            importance_sampling_level: Level of importance sampling ("token" or "sequence")
            
        Returns:
            Tuple of (loss tensor, average reward)
        """

        if importance_sampling_level == "sequence":
            assert loss_type == "gspo"
       
        # Compute current token log probabilities
        token_log_probs = self._compute_log_probs(
            self.model,
            rollout_data["input_ids"],
            rollout_data["attention_mask"],
            rollout_data["logits_to_keep"]
        )
        
        # Calculate probability ratio
        if importance_sampling_level == "token":
            ratio = torch.exp(token_log_probs - rollout_data["old_log_probs"])
        else:  
            # sequence-level importance sampling for GSPO   
            # Calculate the sequence-level log ratio per paper's formula (7).
            # GSPO's objective is sequence-level. However, by broadcasting the single
            # sequence-level ratio (seq_ratio) to every token in the sequence,
            # we can reuse the token-level PPO loss computation framework.
            # The resulting gradient is mathematically equivalent to a sequence-level objective
            # because all tokens within a sequence share the same advantage and the same ratio.         
            
            # 计算序列级的 log ratio
            log_ratio = token_log_probs - rollout_data["old_log_probs"]
            
            # 1. 使用 completion_mask 屏蔽 padding tokens
            # 2. 沿序列维度求和，然后除以序列的实际长度
            sum_log_ratio = (log_ratio * rollout_data["completion_mask"]).sum(dim=1)
            seq_len = rollout_data["completion_mask"].sum(dim=1).clamp(min=1.0) # clamp 避免除零
            seq_ratio = torch.exp(sum_log_ratio / seq_len)
            
            # 将序列级的 ratio 广播到每个 token 上
            ratio = seq_ratio.unsqueeze(-1).expand_as(token_log_probs)
        
        # Compute rewards
        rewards = torch.tensor(
            self._combined_reward(
                rollout_data["repeated_prompts"],
                rollout_data["formatted_completions"],
                rollout_data["repeated_answers"]
            ),
            dtype=torch.float32,
            device=self.device
        )
        
        # Reshape rewards and calculate advantages
        batch_size = rollout_data["batch_size"]
        num_generations = rollout_data["num_generations"]
        rewards = rewards.view(batch_size, num_generations)
        
        avg_reward = rewards.mean().item()
        logger.info(f"Average Reward: {avg_reward:.6f}")
        
        # Calculate advantages based on loss type
        if loss_type == "grpo" or loss_type == "gspo":
            # Original GRPO: standardize rewards within each prompt
            # GSPO: https://arxiv.org/pdf/2507.18071
            mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
            std_rewards = rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
            std_rewards = torch.clamp(std_rewards, min=1e-4)  # Avoid division by zero
            advantages = ((rewards.view(-1) - mean_rewards) / std_rewards).unsqueeze(1)
        elif loss_type == "dr_grpo":
            # DrGRPO: remove standardization, use reward difference directly
            mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
            advantages = (rewards.view(-1) - mean_rewards).unsqueeze(1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Compute PPO surrogate objective with clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        surrogate_loss = torch.min(surr1, surr2)
        
        # Compute KL divergence
        kl = torch.exp(rollout_data["ref_log_probs"] - token_log_probs) - \
             (rollout_data["ref_log_probs"] - token_log_probs) - 1
        
        # Combine losses
        per_token_loss = surrogate_loss - beta * kl
        loss = -((per_token_loss * rollout_data["completion_mask"]).sum(dim=1) / 
                rollout_data["completion_mask"].sum(dim=1)).mean()
        
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
        importance_sampling_level: str  # Can be "token" or "sequence"
    ):
        """
        Train the model using RLVR algorithm (e.g. GRPO, DrGRPO, GSPO).
        
        Args:
            train_data: Training dataset
            num_iterations: Number of outer iterations (reference model updates)
            num_steps: Number of batch updates per iteration
            batch_size: Number of prompts per batch
            num_generations: Number of completions per prompt
            max_completion_length: Maximum token length for completions
            beta: KL penalty coefficient
            learning_rate: Learning rate for optimizer
            epsilon: PPO clipping parameter
            loss_type: Type of loss to use ("grpo", "dr_grpo", or "gspo")
            importance_sampling_level: Level of importance sampling ("token" or "sequence")
        """
        logger.info("\nStarting RL training...")
        logger.info(f"Using algorithm: {loss_type.upper()} with {importance_sampling_level}-level importance sampling")
        
        for iteration in range(num_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"ITERATION {iteration+1}/{num_iterations}")
            logger.info(f"{'='*50}")
            
            # Create reference model
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("Reference model created")
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            self.model.train()
            
            # Training loop
            for step in range(num_steps):
                # Sample batch
                batch_samples = random.sample(train_data, batch_size)
                
                # Generate rollout data
                rollout_data = self._generate_rollout_data(
                    batch_samples,
                    num_generations,
                    max_completion_length
                )
                
                # Policy updates
                loss, avg_reward = self._compute_loss(
                    rollout_data,
                    beta=beta,
                    epsilon=epsilon,
                    loss_type=loss_type,
                    importance_sampling_level=importance_sampling_level
                )
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                optimizer.step()
                
                # Log progress
                logger.info(
                    f"Iter: {iteration+1}/{num_iterations}, "
                    f"Step: {step+1}/{num_steps}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Avg Reward: {avg_reward:.6f}, "
                    f"Algorithm: {loss_type.upper()}"
                )
        
        # Save the model
        logger.info("\nSaving fine-tuned model...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

    def test_model(self, prompts: List[str], max_new_tokens: int = 400):
        """Test the fine-tuned model on example prompts."""
        self.model.eval()
        logger.info("\n" + "="*50)
        logger.info("MODEL TESTING")
        logger.info("="*50)
        
        for i, prompt in enumerate(prompts):
            # Prepare the prompt
            test_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            test_prompt = self._build_prompt(test_messages)
            
            # Generate response
            inputs = self.tokenizer(
                test_prompt, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left"
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
                    use_cache=True # TODO: remove this?
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            extracted_answer = self.extract_answer_from_model_output(response)
            
            # Log results
            
            logger.info(
                f"Test #{i+1} | Prompt: {prompt}\n\n"
                f"Full completion: {response[len(test_prompt):]}\n\n"
                f"Extracted: {extracted_answer}\n"
                + "-"*50
            )
        
        self.model.train()


def main():
    """Main execution function."""

    # Algorithm configuration
    algorithms = [
        ("gspo", "sequence", "GSPO"),
        ("grpo", "token", "GRPO"),
        ("dr_grpo", "token", "DrGRPO"),
    ]

    setup_logging("./logs/")
    
    # Load dataset once to ensure consistency across experiments
    all_data = RLVRTrainer.load_dataset("train")
    random.shuffle(all_data)
    
    # Split into train and eval, default is 50
    eval_size = 50
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    
    # Training configuration
    training_config = {
        'num_iterations': 1,
        'num_steps': 200,
        'batch_size': 2,
        'num_generations': 8,
        'max_completion_length': 400,
        'beta': 0.04,
        'learning_rate': 5e-6,
        'epsilon': 0.1,
    }

    # Initial evaluation    
    dummy_trainer = RLVRTrainer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="raw_model",
    )
    dummy_trainer.setup_model()
    logger.info("\nInitial evaluation for raw model Qwen2.5-1.5B-Instruct ...")
    initial_acc = dummy_trainer.evaluate(eval_data)
    logger.info(f"\nInitial evaluation for raw model Qwen2.5-1.5B-Instruct, accuracy: {initial_acc:.2f}%")

    # Run experiments for each algorithm
    results = {}
    for loss_type, is_level, algo_name in algorithms:
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING EXPERIMENT: {algo_name}")
        logger.info(f"{'='*70}")
        
        # Create a new trainer for each algorithm
        trainer = RLVRTrainer(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir=f"finetuned_model_{loss_type}",
        )
        
        trainer.setup_model()
        trainer.optimize_memory()
        
        # Train with current algorithm
        logger.info(f"\nTraining with {algo_name}...")
        trainer.train(
            train_data, 
            loss_type=loss_type,
            importance_sampling_level=is_level,
            **training_config
        )
        
        # Final evaluation
        logger.info(f"\nFinal evaluation for {algo_name}...")
        final_acc = trainer.evaluate(eval_data)
        logger.info(f"\nFinal evaluation for {algo_name}, accuracy: {final_acc:.2f}%")
        
        # Save results
        results[algo_name] = {
            "initial_accuracy": initial_acc,
            "final_accuracy": final_acc
        }
        
        # Test model
        test_prompts = [
            "How much is 1+1?",
            "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
            "Solve the equation 6x + 4 = 40"
        ]
        trainer.test_model(test_prompts)
    
    # Print comparison results
    logger.info(
        f"{'='*20} EXPERIMENT COMPARISON RESULTS {'='*20}\n" +
        "\n".join(
            f"{algo}: Init {m['initial_accuracy']:.2f}% → Final {m['final_accuracy']:.2f}% "
            f"(Δ {m['final_accuracy'] - m['initial_accuracy']:.2f}%)"
            for algo, m in results.items()
        ) +
        "\n\nExperiment completed!"
    )


if __name__ == "__main__":
    main()