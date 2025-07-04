#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用自定义模型评估 LayoutReader (包含类别信息)。

用法示例：
    python v4/evaluate_custom_model.py \
        --data Data/dev_for_eval.jsonl \
        --model_name /path/to/your/checkpoint/checkpoint-xxxx
"""
import sys
sys.path.append('/workspace/paddlex/layoutreader')  # 或者写成 os.path.abspath(os.path.dirname(...))

import argparse
import json
from typing import List

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

# ============ 关键修改: 导入自定义模型和数据相关模块 ============
from v4.modeling_custom import CustomLayoutLMv3ForTokenClassification
from v4.helpers import LABEL_MAP

# ============ LayoutReader 相关常量与函数 (已更新) ============
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2

def prepare_inputs(inputs, model):
    """将输入张量移动到模型所在的设备上。"""
    return {k: v.to(model.device) for k, v in inputs.items()}

def boxes_and_categories_to_inputs(boxes: List[List[int]], category_types: List[str]):
    """
    将 bbox 和类别标签列表打包为自定义 LayoutLMv3 的输入。
    """
    # 转换类别标签为ID
    category_ids = [LABEL_MAP.get(label, LABEL_MAP["unknown"]) for label in category_types]
    # 为 [CLS] 和 [EOS] 添加占位符
    final_boxes = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    final_category_ids = [0] + category_ids + [0] # 0 for CLS, EOS, PAD
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] * len(input_ids)
    
    return {
        "bbox": torch.tensor([final_boxes], dtype=torch.long),
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "category_ids": torch.tensor([final_category_ids], dtype=torch.long),
    }

def parse_logits(logits: torch.Tensor, length: int) -> List[int]:
    logits = logits[1:length+1, :length]
    orders = logits.argsort(descending=False).tolist()
    pred = [o.pop() for o in orders]

    while True:
        seen = {}
        for i, o in enumerate(pred):
            seen.setdefault(o, []).append(i)
        conflicts = {k: v for k, v in seen.items() if len(v) > 1}
        if not conflicts:
            break
        for _, idxs in conflicts.items():
            for idx in idxs[1:]:
                pred[idx] = orders[idx].pop()
    return pred

# ============ 通用评估函数 (无变化) ============
def edit_distance(a: List[int], b: List[int]) -> int:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                         dp[i][j - 1] + 1,        # Insertion
                         dp[i - 1][j - 1] + cost) # Substitution
    return int(dp[n][m])

# ============ 主评估流程 (已更新) ============
def evaluate_layoutreader(model_name: str, data_file: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"正在加载自定义模型 '{model_name}' 到设备 '{device}'...")
    
    # ============ 关键修改: 加载自定义模型 ============
    model = CustomLayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True # 必须设置，以忽略新增的 embedding 层
    )
    model.eval().to(device)

    stats = {"exact": 0, "kendall": [], "spearman": [], "edit": [], "omni": []}
    total_samples = 0

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)

    for line in tqdm(lines, total=total_lines, desc="正在评估自定义模型"):
        sample = json.loads(line)
        boxes = sample["source_boxes"]
        category_types = sample.get("category_types", [])
        gt_order = sample.get("gt_order")

        if not gt_order or len(boxes) != len(gt_order) or len(boxes) != len(category_types):
            continue

        total_samples += 1
        upper_len = len(gt_order)

        # 构造输入并推理
        with torch.no_grad():
            # ============ 关键修改: 使用新的输入构造函数 ============
            inputs = boxes_and_categories_to_inputs(boxes, category_types)
            inputs = prepare_inputs(inputs, model)
            logits = model(**inputs).logits.squeeze(0).cpu()

        pred_order = parse_logits(logits, len(boxes))
        
        # 检查预测长度是否匹配
        if len(pred_order) != len(gt_order):
            continue

        stats["exact"] += int(pred_order == gt_order)
        stats["kendall"].append(kendalltau(gt_order, pred_order).correlation)
        stats["spearman"].append(spearmanr(gt_order, pred_order).correlation)
        d = edit_distance(gt_order, pred_order)
        stats["edit"].append(d)
        stats["omni"].append(d / upper_len if upper_len > 0 else 0)

    # 汇总结果
    exact = stats["exact"] / total_samples if total_samples > 0 else 0
    kendall = np.nanmean(stats["kendall"]) if stats["kendall"] else 0
    spearman = np.nanmean(stats["spearman"]) if stats["spearman"] else 0
    edit = np.mean(stats["edit"]) if stats["edit"] else 0
    omni = np.mean(stats["omni"]) if stats["omni"] else 0

    print("\n" + "=" * 30 + " 评估结果 (自定义模型) " + "=" * 30)
    print(f"有效样本数: {total_samples}")
    print(f"精确匹配率 (Exact Match): {exact:.4f}")
    print(f"平均 Kendall Tau:        {kendall:.4f}")
    print(f"平均 Spearman Rho:       {spearman:.4f}")
    print(f"平均编辑距离 (Edit Dist):  {edit:.2f}")
    print(f"OmniDocBench 分数 : {omni:.4f}") # 1 - Normalized Edit Distance
    print("=" * 75 + "\n")

# ============ CLI ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用自定义模型评估阅读顺序")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="/workspace/paddlex/layoutreader/checkpoint/v3/2025-07-04-09/checkpoint-1392",
        help="你训练好的自定义模型的检查点路径 (例如: .../checkpoint-3683)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="/workspace/paddlex/test_for_eval_shuf.jsonl",
        help="经过格式转换后的评估数据路径"
    )
    args = parser.parse_args()
    
    # 确保可以找到 v4 模块
    import sys
    sys.path.append('.')
    from loguru import logger
    
    evaluate_layoutreader(args.model_name, args.data)
