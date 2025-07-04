import torch
import random
from typing import List, Dict
from collections import defaultdict
from transformers import LayoutLMv3ForTokenClassification

# 定义常量
MAX_LEN = 510
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2

import torch
import random
from typing import List, Dict
from collections import defaultdict
from transformers import LayoutLMv3ForTokenClassification

# 定义常量
MAX_LEN = 510
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2

class DataCollator:
    """
    一个功能强大的数据整理器，在训练时动态应用数据增强。
    - 顺序增强：按一定概率随机打乱文本框的顺序。
    - 几何增强：对文本框坐标添加随机噪声。
    """
    def __init__(
        self,
        is_training: bool = True,
        shuffle_probability: float = 0.5,
        bbox_noise_level: float = 0.02,
    ):
        """
        初始化数据整理器。

        Args:
            is_training (bool): 是否为训练模式。数据增强仅在训练时应用。
            shuffle_probability (float): 对每个样本的输入框顺序进行随机打乱的概率。
            bbox_noise_level (float): 边界框坐标噪声的强度。
                                      例如0.02表示噪声范围为框宽/高的 ±2%。设为0则关闭。
        """
        self.is_training = is_training
        self.shuffle_probability = shuffle_probability
        self.bbox_noise_level = bbox_noise_level
        random.seed(42)

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        bbox_list, labels_list, input_ids_list, attention_mask_list = [], [], [], []

        for feature in features:
            source_boxes = feature["source_boxes"]
            target_indexes = feature["target_index"]

            _bbox = source_boxes
            _labels = target_indexes

            # --- 应用数据增强（仅在训练时）---
            if self.is_training:
                # 这里的 target_indexes 已经是我们需要的标签列表了
                _labels = target_indexes 
                if self.is_training and random.random() < self.shuffle_probability:
                    # ① 生成一次新的随机排列 perm
                    perm = list(range(len(source_boxes)))   # [0, 1, 2, ... , N-1]
                    random.shuffle(perm)

                    # ② 用 perm 重排 source_boxes  →  得到最终送进模型的 _bbox
                    _bbox = [source_boxes[i] for i in perm]

                    # ③ 根据 perm **重新构造 target_index**
                    #
                    #    new_target_index[orig_idx] = new_pos + 1
                    #    ───────────┬───  ──────┬─
                    #               │          │
                    #   阅读顺序里的编号   现在在 _bbox 里的位置 (1-based)
                    #
                    new_target_index = [0] * len(perm)
                    for new_pos, orig_idx in enumerate(perm):
                        new_target_index[orig_idx] = new_pos + 1   # +1 保留 0 给 ignore

                    _labels = new_target_index                    # 这才是新的标签
                else:
                    _bbox   = source_boxes[:]                     # 不打乱就直接用
                    _labels = target_indexes[:]                   # label 也保持原样

                    _bbox = source_boxes
                # 2. 边界框坐标噪声
                if self.bbox_noise_level > 0:
                    noisy_bbox = []
                    for box in _bbox:
                        w, h = box[2] - box[0], box[3] - box[1]
                        
                        noise_x = w * self.bbox_noise_level * (random.random() * 2 - 1)
                        noise_y = h * self.bbox_noise_level * (random.random() * 2 - 1)
                        
                        new_box = [
                            max(0, min(1000, round(box[0] + noise_x))),
                            max(0, min(1000, round(box[1] + noise_y))),
                            max(0, min(1000, round(box[2] + noise_x))),
                            max(0, min(1000, round(box[3] + noise_y))),
                        ]
                        
                        if new_box[0] >= new_box[2]: new_box[2] = new_box[0] + 1
                        if new_box[1] >= new_box[3]: new_box[3] = new_box[1] + 1
                        
                        noisy_bbox.append(new_box)
                    _bbox = noisy_bbox

            # --- 截断、构建ID和掩码 ---
            if len(_bbox) > MAX_LEN:
                _bbox, _labels = _bbox[:MAX_LEN], _labels[:MAX_LEN]

            _input_ids = [UNK_TOKEN_ID] * len(_bbox)
            _attention_mask = [1] * len(_bbox)
            
            bbox_list.append(_bbox)
            labels_list.append(_labels)
            input_ids_list.append(_input_ids)
            attention_mask_list.append(_attention_mask)
        
        # --- 添加特殊符并进行填充 ---
        # 先添加 [CLS] 和 [EOS]
        for i in range(len(bbox_list)):
            bbox_list[i] = [[0, 0, 0, 0]] + bbox_list[i] + [[0, 0, 0, 0]]
            labels_list[i] = [-100] + labels_list[i] + [-100]
            input_ids_list[i] = [CLS_TOKEN_ID] + input_ids_list[i] + [EOS_TOKEN_ID]
            attention_mask_list[i] = [1] + attention_mask_list[i] + [1]

        # 找到添加特殊符后的最大长度
        max_len_in_batch = max(len(x) for x in bbox_list)

        # 再进行填充
        for i in range(len(bbox_list)):
            pad_len = max_len_in_batch - len(bbox_list[i])
            bbox_list[i] += [[0, 0, 0, 0]] * pad_len
            labels_list[i] += [-100] * pad_len
            input_ids_list[i] += [EOS_TOKEN_ID] * pad_len
            attention_mask_list[i] += [0] * pad_len

        # --- 转换为 Tensor ---
        ret = {
            "bbox": torch.tensor(bbox_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        }
        
        # --- 后处理标签 ---
        ret["labels"][ret["labels"] > MAX_LEN] = -100
        ret["labels"][ret["labels"] > 0] -= 1
        
        return ret


def parse_logits(logits: torch.Tensor, length: int) -> List[int]:
    """
    parse logits to orders

    :param logits: logits from model
    :param length: input length
    :return: orders
    """
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        # filter idxes len > 1
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        # filter
        for order, idxes in order_to_idxes.items():
            # find original logits of idxes
            idxes_to_logit = {}
            for idx in idxes:
                idxes_to_logit[idx] = logits[idx, order]
            idxes_to_logit = sorted(
                idxes_to_logit.items(), key=lambda x: x[1], reverse=True
            )
            # keep the highest logit as order, set others to next candidate
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()

    return ret


def check_duplicate(a: List[int]) -> bool:
    return len(a) != len(set(a))
