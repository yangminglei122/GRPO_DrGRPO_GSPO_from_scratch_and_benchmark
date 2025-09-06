from pathlib import Path
from trainer import *


def main():
    """Main execution function."""

    # 算法参数配置
    algorithms = [
        ("gspo", "sequence", "GSPO"),
        ("grpo", "token", "GRPO"),
        ("dr_grpo", "token", "DrGRPO"),
    ]

    # 设置日志
    logger = setup_logging("./logs/")

    # 加载数据集，确保实验一致性
    all_data = RLVRTrainer.load_dataset("train")
    random.shuffle(all_data)

    # 划分训练集和验证集，默认50%
    eval_size = 50
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]

    # 训练参数配置
    training_config = {
        "num_iterations": 1,
        "num_steps": 200,
        "batch_size": 1,
        "num_generations": 8,
        "max_completion_length": 400,
        "beta": 0.04,
        "learning_rate": 5e-6,
        "epsilon": 0.1,
    }

    # model 路径
    model_origin_dir = Path("/home/yangml/Models/Qwen3-0.6B/")
    model_output_dir = Path("/home/yangml/Models/Qwen3-0.6B-compare/")

    # Initial evaluation
    dummy_trainer = RLVRTrainer(
        model_name=model_origin_dir,  # "Qwen/Qwen3-0.6B", # 考虑显卡
        output_dir=model_output_dir / "raw_model",
    )

    dummy_trainer.setup_model()
    print("\nInitial evaluation for raw model Qwen3-0.6B ...")
    initial_acc = dummy_trainer.evaluate(eval_data)
    logger.info(
        f"\nInitial evaluation for raw model Qwen3-0.6B, accuracy: {initial_acc:.2f}%"
    )

    # 每种算法都执行训练过程
    results = {}
    for loss_type, is_level, algo_name in algorithms:
        logger.info(f"\n{'='*70}")
        logger.info(f"开始训练 {algo_name}: {algo_name}")
        logger.info(f"{'='*70}")

        # Create a new trainer for each algorithm
        trainer = RLVRTrainer(
            model_name=model_origin_dir,  # "Qwen/Qwen3-0.6B",
            output_dir=model_output_dir / f"finetuned_model_{loss_type}",
        )

        trainer.setup_model()
        trainer.optimize_memory()

        # 训练当前算法
        logger.info(f"\n训练-{algo_name}...")
        trainer.train(
            train_data,
            loss_type=loss_type,
            importance_sampling_level=is_level,
            **training_config,
        )

        # Final evaluation
        logger.info(f"\n最终评估： {algo_name}...")
        final_acc = trainer.evaluate(eval_data)
        logger.info(f"\n最终评估结果： {algo_name}, accuracy: {final_acc:.2f}%")

        # Save results
        results[algo_name] = {
            "initial_accuracy": initial_acc,
            "final_accuracy": final_acc,
        }

        # Test model
        test_prompts = [
            "How much is 1+1?",
            "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
            "Solve the equation 6x + 4 = 40",
        ]
        trainer.test_model(test_prompts)

    # 打印对比结果
    logger.info(
        f"{'='*20} 对比实验结果 {'='*20}\n"
        + "\n".join(
            f"{algo}: Init {m['initial_accuracy']:.2f}% → Final {m['final_accuracy']:.2f}% "
            f"(Δ {m['final_accuracy'] - m['initial_accuracy']:.2f}%)"
            for algo, m in results.items()
        )
        + "\n\n实验结束!"
    )


if __name__ == "__main__":
    main()
