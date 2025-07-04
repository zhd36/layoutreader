# v4/modeling_custom.py

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers.models.layoutlmv3.modeling_layoutlmv3 import (
    LayoutLMv3TextEmbeddings,
    LayoutLMv3Model,
    LayoutLMv3ForTokenClassification,
    TokenClassifierOutput,
)
from transformers.modeling_outputs import BaseModelOutput

# 你的类别数量
NUM_CATEGORIES = 50

# ==================== DEBUG FLAG ====================
# 用于控制调试信息只打印一次的全局标志
DEBUG_PRINTED = False
# ====================================================

# 1. 继承并扩展 TextEmbeddings (无变化)
class CustomLayoutLMv3TextEmbeddings(LayoutLMv3TextEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.category_embeddings = nn.Embedding(NUM_CATEGORIES, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        category_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)
        embeddings += spatial_position_embeddings

        if category_ids is not None:
            category_embedding_output = self.category_embeddings(category_ids)
            embeddings += category_embedding_output

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# 2. 继承并扩展核心模型 (无变化)
class CustomLayoutLMv3Model(LayoutLMv3Model):
    def __init__(self, config):
        super().__init__(config)
        if config.text_embed:
            self.embeddings = CustomLayoutLMv3TextEmbeddings(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        category_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        embedding_output = None
        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                category_ids=category_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = bbox
        if position_ids is None and (input_ids is not None or inputs_embeds is not None):
             if input_ids is not None:
                 position_ids = self.embeddings.create_position_ids_from_input_ids(input_ids, self.embeddings.padding_idx).to(input_ids.device)
             else:
                 position_ids = self.embeddings.create_position_ids_from_inputs_embeds(inputs_embeds)
        final_position_ids = position_ids
        
        if pixel_values is not None:
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones((batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                visual_position_ids = torch.arange(0, visual_embeddings.shape[1], dtype=torch.long, device=device).repeat(batch_size, 1)
                if embedding_output is not None:
                     final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                     final_position_ids = visual_position_ids
            
            if embedding_output is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings
            
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, None, device, dtype=embedding_output.dtype)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# 3. 继承并组装最终的分类模型 (包含调试代码)
class CustomLayoutLMv3ForTokenClassification(LayoutLMv3ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlmv3 = CustomLayoutLMv3Model(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        category_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        
        global DEBUG_PRINTED # 使用全局标志

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            category_ids=category_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # ==================== DEBUG CODE START ====================
            # 只在第一次遇到带标签的批次时打印 (通常在评估或训练时)
            if not DEBUG_PRINTED and not self.training:
                print("\n" + "="*25 + " DEBUG OUTPUT (FIRST EVAL BATCH) " + "="*25)
                
                # 获取批次中的第一个样本
                first_logits = logits[0]
                first_labels = labels[0]
                
                # 找到有效长度 (排除-100和填充)
                # CLS token is at index 0, so we start from 1
                valid_indices = (first_labels[1:] != -100).nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    # add 1 to get length, add 1 to account for CLS token
                    valid_length = valid_indices[-1].item() + 2 
                    
                    # 真实标签 (排除了 CLS, EOS, 和 padding)
                    # +1 是因为原始的 target_index 是从1开始的，而 DataCollator 中减了1
                    gt_order = (first_labels[1:valid_length-1] + 1).tolist()

                    # 预测结果
                    pred_logits = first_logits[1:valid_length-1]
                    pred_order_idx = torch.argmax(pred_logits, dim=-1).tolist()
                    
                    print(f"Sample 0 in batch | Valid Length (excluding CLS/EOS): {len(gt_order)}")
                    print(f"  -> Ground Truth Order : {gt_order}")
                    print(f"  -> Predicted Index    : {pred_order_idx}")

                print("="*81 + "\n")
                DEBUG_PRINTED = True # 确保不再打印
            # ===================== DEBUG CODE END =====================


        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
