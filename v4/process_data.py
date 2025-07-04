import gzip
import json
import os
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from loguru import logger


def convert_dataset(
    data: Dict,
    output_file: str,
    shuffle_rate: float = 0.5,
):
    """
    将数据列表保存为 jsonl.gz 格式，构建 source/target boxes 和 labels。
    """
    random.seed(42)
    with gzip.open(output_file, "wt", encoding="utf-8") as f_out:
        for image_name, image_data in tqdm(data.items()):
            width = image_data.get("image_size", {}).get("width", 0)
            height = image_data.get("image_size", {}).get("height", 0)
            blocks: List[dict] = image_data.get("image_info", [])

            if width == 0 or height == 0 or not isinstance(blocks, list):
                logger.warning(f"跳过无效图片：{image_name}")
                continue

            try:
                sorted_blocks = sorted(blocks, key=lambda x: x["order_id"])
            except Exception:
                logger.warning(f"排序失败：{image_name}")
                continue

            target_boxes = []
            target_labels = []

            for block in sorted_blocks:
                if "block_bbox" not in block:
                    continue

                bbox = block["block_bbox"]
                label = block.get("block_label", "unknown")

                try:
                    norm_left = round(bbox[0] * 1000 / width)
                    norm_top = round(bbox[1] * 1000 / height)
                    norm_right = round(bbox[2] * 1000 / width)
                    norm_bottom = round(bbox[3] * 1000 / height)
                except Exception:
                    continue

                norm_left = max(0, min(1000, norm_left))
                norm_top = max(0, min(1000, norm_top))
                norm_right = max(0, min(1000, norm_right))
                norm_bottom = max(0, min(1000, norm_bottom))

                if norm_right < norm_left or norm_bottom < norm_top:
                    continue

                target_boxes.append([norm_left, norm_top, norm_right, norm_bottom])
                target_labels.append(label)

            if not target_boxes:
                logger.warning(f"跳过空文本块：{image_name}")
                continue

            index_and_box = list(enumerate(target_boxes))
            if random.random() < shuffle_rate:
                random.shuffle(index_and_box)
            else:
                #index_and_box.sort(key=lambda x: (x[1][1], x[1][0]))
                index_and_box = list(enumerate(target_boxes))  # 原始顺序不动


            source_boxes = []
            source_labels = []
            target_index = [0] * len(target_boxes)

            for new_idx, (orig_idx, _) in enumerate(index_and_box):
                source_boxes.append(target_boxes[orig_idx])
                source_labels.append(target_labels[orig_idx])
                target_index[orig_idx] = new_idx + 1

            record = {
                "source_boxes": source_boxes,
                "source_labels": source_labels,
                "target_boxes": target_boxes,
                "target_labels": target_labels,
                "target_index": target_index,
                "image_name": image_name,
                "bleu": 0.0
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.success(f"保存完成：{output_file}")


def split_dataset(
    data: Dict,
    train_ratio=0.8,
    dev_ratio=0.1,
) -> Tuple[Dict, Dict, Dict]:
    """
    按比例划分数据为 train/dev/test
    """
    keys = list(data.keys())
    random.shuffle(keys)

    total = len(keys)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_keys = keys[:train_end]
    dev_keys = keys[train_end:dev_end]
    test_keys = keys[dev_end:]

    train_data = {k: data[k] for k in train_keys}
    dev_data = {k: data[k] for k in dev_keys}
    test_data = {k: data[k] for k in test_keys}

    logger.info(f"训练集: {len(train_data)}，验证集: {len(dev_data)}，测试集: {len(test_data)}")

    return train_data, dev_data, test_data


def main():
    input_path = "merge_data_0703.json"
    train_output = "/workspace/paddlex/layoutreader/Data/train.jsonl.gz"
    dev_output = "/workspace/paddlex/layoutreader/Data/dev.jsonl.gz"
    test_output = "/workspace/paddlex/layoutreader/Data/test.jsonl.gz"
    test_shuf_output = "/workspace/paddlex/layoutreader/Data/test_shuf.jsonl.gz"

    # 加载整体数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 数据划分
    train_data, dev_data, test_data = split_dataset(data)

    # 转换为训练数据格式
    convert_dataset(train_data, train_output, shuffle_rate=0)
    convert_dataset(dev_data, dev_output, shuffle_rate=0)
    convert_dataset(test_data, test_output, shuffle_rate=0.0)
    convert_dataset(test_data, test_shuf_output, shuffle_rate=1.0)


if __name__ == "__main__":
    main()
