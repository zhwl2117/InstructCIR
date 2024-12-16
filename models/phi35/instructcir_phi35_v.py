import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoConfig, LlavaConfig
from models.phi35.modeling_llava import Phi3LlavaForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def concat_all_gather(tensor):
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class InstructCIRLlavaPhi35Config(LlavaConfig):
    model_type = "instructcir_llava_phi35"


class InstructCIRLlavaPhi35ForConditionalGeneration(Phi3LlavaForConditionalGeneration):
    config_class = InstructCIRLlavaPhi35Config

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "ret_proj_dim", None):
            self.ret_proj = nn.Linear(config.hidden_size, config.ret_proj_dim)
        else:
            self.ret_proj = None #  wait to be initialized later
        # self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(1 / 0.07))

        self.post_init()

    def compute_contrast_loss(self, z1, z2, z3):
        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z3_list[dist.get_rank()] = z3
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        z3 = torch.cat(z3_list, 0)

        cos_sim = F.cosine_similarity(z1.unsqueeze(1).float(), z2.unsqueeze(0).float(), dim=-1) / 0.05

        z1_z3_cos = F.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0), dim=-1) / 0.05
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        z3_weight = 0
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(z1.device)
        cos_sim = cos_sim + weights

        contrast_loss = F.cross_entropy(cos_sim, labels)
        return contrast_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        conditioned_input_ids: torch.LongTensor = None,
        conditioned_attention_mask: Optional[torch.Tensor] = None,
        pos_original_input_ids: torch.LongTensor = None,
        pos_original_attention_mask: Optional[torch.Tensor] = None,
        pos_conditioned_input_ids: torch.LongTensor = None,
        pos_conditioned_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        # original_outputs = super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=True,
        #     pixel_values=pixel_values,
        #     return_dict=return_dict,
        # )

        # hidden_state = original_outputs.hidden_states[-1]
        # original_embeddings = hidden_state[:, -1, :]

        conditioned_outputs = super().forward(
            input_ids=conditioned_input_ids,
            attention_mask=conditioned_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            pixel_values=pixel_values,
            return_dict=return_dict,
        )

        hidden_state = conditioned_outputs.hidden_states[-1]
        conditioned_embeddings = hidden_state[:, -1, :]

        pos_original_outputs = super().forward(
            input_ids=pos_original_input_ids,
            attention_mask=pos_original_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            pixel_values=None,
            return_dict=return_dict,
        )

        hidden_state = pos_original_outputs.hidden_states[-1]
        pos_original_embeddings = hidden_state[:, -1, :]

        pos_conditioned_outputs = super().forward(
            input_ids=pos_conditioned_input_ids,
            attention_mask=pos_conditioned_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            pixel_values=None,
            return_dict=return_dict,
        )

        hidden_state = pos_conditioned_outputs.hidden_states[-1]
        pos_conditioned_embeddings = hidden_state[:, -1, :]

        # z1, z2, z3 = original_embeddings, pos_original_embeddings, pos_conditioned_embeddings
        # original_contrast_loss = self.compute_contrast_loss(z1, z2, z3)

        z1, z2, z3 = conditioned_embeddings, pos_conditioned_embeddings, pos_original_embeddings
        conditioned_contrast_loss = self.compute_contrast_loss(z1, z2, z3)

        # contrast_loss = (original_contrast_loss + conditioned_contrast_loss) / 2
        contrast_loss = conditioned_contrast_loss

        return CausalLMOutputWithPast(
            loss=contrast_loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def encode(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,  # to align with the setting of phi3-vision
        return_dict: Optional[bool] = None,
        last_token_only: bool = True,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            return_dict=return_dict,
        )

        hidden_state = outputs.hidden_states[-1]
        if last_token_only:
            ret_embeddings = hidden_state[:, -1, :]
            ret_embeddings = F.normalize(ret_embeddings, dim=-1, p=2)
            return ret_embeddings
        else:
            return hidden_state
