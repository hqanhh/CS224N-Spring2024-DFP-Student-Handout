import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *
import math

class FlashAttention(nn.Module):
    def __init__(self, head_size):
        super(FlashAttention, self).__init__()
        self.head_size = head_size

    def forward(self, query, key, value, attention_mask):
        B, nh, T, hs = query.size()
        # Efficient attention computation
        attn_weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ value
        return attn_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # FlashAttention initialization
        self.flash_attention = FlashAttention(self.attention_head_size)

    def transform(self, x, linear_layer):
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        attn_output = self.flash_attention(query, key, value, attention_mask)
        B, nh, T, hs = query.size()
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, nh * hs)
        return attn_output

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        output = dense_layer(output)
        output = dropout(output)
        output = ln_layer(output + input)
        return output

    def forward(self, hidden_states, attention_mask):
        attn_output = self.self_attention(hidden_states, attention_mask)
        add_norm_output = self.add_norm(hidden_states, attn_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
        feed_forward_output = self.interm_af(self.interm_dense(add_norm_output))
        add_norm_output = self.add_norm(add_norm_output, feed_forward_output, self.out_dense, self.out_dropout, self.out_layer_norm)
        return add_norm_output

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)
        embeds_agr = inputs_embeds + pos_embeds + tk_type_embeds
        embeds_agr = self.embed_dropout(self.embed_layer_norm(embeds_agr))
        return embeds_agr

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
        for i, layer_module in enumerate(self.bert_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embed(input_ids=input_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)
        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
