import os
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    LayoutLMv3ForTokenClassification,
    set_seed,
)
from transformers.trainer import Trainer

from helpers import DataCollator, MAX_LEN


import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Arguments(TrainingArguments):
    model_dir: str = field(
        default=None,
        metadata={"help": "模型路径, 基于 `microsoft/layoutlmv3-base`"},
    )
    dataset_dir: str = field(
        default=None,
        metadata={"help": "数据集路径"},
    )
    shuffle_probability: float = field(
        default=0.5,
        metadata={"help": "为数据增强而打乱阅读顺序的概率。"},
    )
    # 我们可以为噪声水平也添加一个参数
    bbox_noise_level: float = field(
        default=0.02,
        metadata={"help": "为数据增强添加的坐标噪声强度。"}
    )

def load_train_and_dev_dataset(path: str) -> (Dataset, Dataset):
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
    """
    def get_train_dataloader(self) :
        """
        返回训练数据加载器。数据增强将被启用。
        """
        # 为训练集创建一个开启数据增强的 DataCollator
        self.data_collator = DataCollator(
            is_training=True,
            shuffle_probability=self.args.shuffle_probability,
            bbox_noise_level=self.args.bbox_noise_level
        )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Dataset = None) :
        """
        返回验证数据加载器。数据增强将被禁用。
        """
        # 为验证集创建一个关闭数据增强的 DataCollator
        self.data_collator = DataCollator(is_training=False)
        return super().get_eval_dataloader(eval_dataset)

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    train_dataset, dev_dataset = load_train_and_dev_dataset(args.dataset_dir)
    logger.info(
        "训练集大小: {}, 验证集大小: {}".format(
            len(train_dataset), len(dev_dataset)
        )
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_dir, num_labels=MAX_LEN, visual_embed=False
    )
    # 将打乱概率传递给 DataCollator
    data_collator = DataCollator(shuffle_probability=args.shuffle_probability)
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
