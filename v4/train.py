# v4/train.py

import os
import warnings
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    Trainer
)

# ==================== 关键修改 (1/2) ====================
# 从我们自定义的文件中导入模型和数据处理器
from modeling_custom import CustomLayoutLMv3ForTokenClassification
from helpers import DataCollator, MAX_LEN
# =========================================================

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Arguments(TrainingArguments):
    model_dir: str = field(
        default=None,
        metadata={"help": "模型路径, 用于加载预训练权重。例如 'microsoft/layoutlmv3-base'"},
    )
    dataset_dir: str = field(
        default=None,
        metadata={"help": "包含 train.jsonl.gz 和 dev.jsonl.gz 的数据集目录。"},
    )
    shuffle_probability: float = field(
        default=0.5,
        metadata={"help": "数据增a强：随机打乱输入顺序的概率。"},
    )
    bbox_noise_level: float = field(
        default=0.02,
        metadata={"help": "数据增强：为边界框坐标添加的噪声强度。"}
    )

def load_train_and_dev_dataset(path: str) -> (Dataset, Dataset):
    """加载训练集和验证集"""
    logger.info(f"从目录 '{path}' 加载数据集...")
    datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(path, "train.jsonl.gz"),
            "dev": os.path.join(path, "dev.jsonl.gz"),
        },
    )
    return datasets["train"], datasets["dev"]

class CustomTrainer(Trainer):
    """
    自定义Trainer，为训练和评估使用不同配置的DataCollator。
    这样可以确保验证集上不会进行数据增强，使得评估结果更稳定。
    """
    def get_train_dataloader(self) :
        """返回训练数据加载器，启用数据增强。"""
        self.data_collator = DataCollator(
            is_training=True,
            shuffle_probability=self.args.shuffle_probability,
            bbox_noise_level=self.args.bbox_noise_level
        )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Dataset = None) :
        """返回验证数据加载器，禁用数据增强。"""
        # is_training=False 会关闭随机打乱和坐标噪声
        self.data_collator = DataCollator(is_training=False)
        return super().get_eval_dataloader(eval_dataset)

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    
    logger.info("设置随机种子: {}", args.seed)
    set_seed(args.seed)

    train_dataset, dev_dataset = load_train_and_dev_dataset(args.dataset_dir)
    logger.info(
        "数据集加载完成。训练集大小: {}, 验证集大小: {}".format(
            len(train_dataset), len(dev_dataset)
        )
    )

    logger.info("加载自定义模型: CustomLayoutLMv3ForTokenClassification")
    # ==================== 关键修改 (2/2) ====================
    model = CustomLayoutLMv3ForTokenClassification.from_pretrained(
        args.model_dir, 
        num_labels=MAX_LEN, 
        visual_embed=False,
        # 忽略我们新增的 category_embeddings 层的权重不匹配警告
        ignore_mismatched_sizes=True 
    )
    # =========================================================

    logger.info("初始化 Trainer...")
    # Trainer 初始化时不需要 data_collator，因为它会通过 get_train/eval_dataloader 方法动态创建
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    
    logger.info("************ 开始训练 ************")
    trainer.train()
    logger.success("************ 训练完成 ************")


if __name__ == "__main__":
    main()
