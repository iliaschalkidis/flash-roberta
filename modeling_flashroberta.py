from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from flash_attn import flash_attn_func
import torch.nn as nn
import torch
from typing import Optional, Tuple


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->FlashRoberta
class FlashRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout_rate = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Flash Attention
        context_layer, _, attention_probs = flash_attn_func(query_layer, key_layer, value_layer,
                                                            dropout_p=self.dropout_rate,
                                                            softmax_scale=None,
                                                            causal=self.is_decoder,
                                                            return_attn_probs=True)

        # Merge heads
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class FlashRobertaModel(RobertaModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in
    *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* by Tri Dao

    .. _*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*: https://tridao.me/publications/flash2/flash2.pdf

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.__init__ with Roberta->FlashRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        # Replace legacy RobertaSelfAttention with FlashRobertaSelfAttention
        for attention_layer in self.encoder.layer:
            attention_layer.attention.self = FlashRobertaSelfAttention(config)

        # Initialize weights and apply final processing
        self.post_init()


class FlashRobertaForMaskedLM(RobertaForMaskedLM):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        # Replace legacy RobertaModel with FlashRobertaModel
        self.roberta = FlashRobertaModel(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()
