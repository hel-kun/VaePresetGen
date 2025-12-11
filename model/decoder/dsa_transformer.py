import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable
from utils.types import *
from utils.synth1_params import *
import math

class DualStreamTransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=2048, 
        dropout=0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu, 
        layer_norm_eps: float = 1e-5, 
        batch_first=True, 
        norm_first: bool = False, 
        bias: bool = True,
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.norm_first = norm_first

        # Masked Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)

        # encoder-decoder attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)

        # cross-stream attention
        self.cross_stream_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout3 = nn.Dropout(dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout4 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout5 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = F.gelu if activation == "gelu" else F.relu
        else:
            self.activation = activation

    def forward(
        self,
        tgt,
        memory,
        cross_stream_memory,
        tgt_mask=None,
        memory_mask=None,
        cross_stream_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        cross_stream_key_padding_mask=None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
            x = x + self._cs_block(
                self.norm3(x), cross_stream_memory, cross_stream_mask, cross_stream_key_padding_mask, memory_is_causal
            )
            x = x + self._ffn_block(self.norm4(x))
        else:
            x = self.norm1(x + self._sa_block(
                x, tgt_mask, tgt_key_padding_mask, tgt_is_causal
            ))
            x = self.norm2(x + self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
            ))
            x = self.norm3(x + self._cs_block(
                x, cross_stream_memory, cross_stream_mask, cross_stream_key_padding_mask, memory_is_causal
            ))
            x = self.norm4(x + self._ffn_block(x))
        return x
    
    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)
    
    def _mha_block(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        x = self.multihead_attn(
            x, memory, memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)
    
    def _cs_block(
        self,
        x: Tensor,
        cross_stream_memory: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        x = self.cross_stream_attn(
            x, cross_stream_memory, cross_stream_memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout3(x)
    
    def _ffn_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout4(self.activation(self.linear1(x))))
        return self.dropout5(x)

class PresetGenDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        categorical_param_size: Optional[dict] = CATEG_PARAM_SIZE
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        num_categ = len(CATEGORICAL_PARAM_NAMES)
        num_cont = len(CONTINUOUS_PARAM_NAMES)
        self.categ_queries = nn.Parameter(torch.randn(len(CATEGORICAL_PARAM_NAMES), embed_dim))
        self.cont_queries = nn.Parameter(torch.randn(len(CONTINUOUS_PARAM_NAMES), embed_dim))
        nn.init.xavier_uniform_(self.categ_queries)
        nn.init.xavier_normal_(self.cont_queries)

        # パラメタ名埋め込みを加算
        self.categ_name_embed = nn.Embedding(num_categ, embed_dim)
        self.cont_name_embed = nn.Embedding(num_cont, embed_dim)

        self.categ_pos_embed = nn.Parameter(torch.zeros(1, num_categ, embed_dim))
        self.cont_pos_embed = nn.Parameter(torch.zeros(1, num_cont, embed_dim))
        nn.init.normal_(self.categ_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.cont_pos_embed, mean=0.0, std=0.02)

        self.categ_name_to_idx = {name: idx for idx, name in enumerate(CATEGORICAL_PARAM_NAMES)}
        self.cont_name_to_idx = {name: idx for idx, name in enumerate(CONTINUOUS_PARAM_NAMES)}

        self.categorical_decoder_layers = nn.ModuleList([
            DualStreamTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        self.continuius_decoder_layers = nn.ModuleList([
            DualStreamTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

        self.continuous_norm = nn.LayerNorm(embed_dim)
        self.categorical_norm = nn.LayerNorm(embed_dim)

        default_categ_sizes = categorical_param_size if categorical_param_size is not None else CATEG_PARAM_SIZE
        self.categorical_param_heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, size) for name, size in default_categ_sizes.items()
        })
        self.continuius_param_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
            for name in CONTINUOUS_PARAM_NAMES
        })

    def forward(
        self,
        memory: Tensor,
        tgt_cont: Optional[Tensor] = None,
        tgt_categ: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tuple[dict, dict]:
        batch_size = memory.size(0)
        categ_query = self.categ_queries.unsqueeze(0).expand(batch_size, -1, -1)
        cont_query = self.cont_queries.unsqueeze(0).expand(batch_size, -1, -1)

        categ_ids = torch.arange(len(CATEGORICAL_PARAM_NAMES), device=memory.device)
        cont_ids = torch.arange(len(CONTINUOUS_PARAM_NAMES), device=memory.device)

        categ_query = categ_query + self.categ_name_embed(categ_ids).unsqueeze(0) + self.categ_pos_embed
        cont_query = cont_query + self.cont_name_embed(cont_ids).unsqueeze(0) + self.cont_pos_embed
        categorical_output = categ_query
        continuous_output = cont_query

        continuous_intermediates = [continuous_output]
        categorical_intermediates = [categorical_output]

        for i in range(self.num_layers):
            categorical_output = self.categorical_decoder_layers[i](
                categorical_output,
                memory,
                continuous_intermediates[-1],
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                cross_stream_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            continuous_output = self.continuius_decoder_layers[i](
                continuous_output,
                memory,
                categorical_intermediates[-1],
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                cross_stream_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

            categorical_intermediates.append(categorical_output)
            continuous_intermediates.append(continuous_output)
        
        # 正規化
        categorical_output = self.categorical_norm(categorical_output)
        continuous_output = self.continuous_norm(continuous_output)

        outputs = {'categorical': {}, 'continuous': {}}

        for name, head in self.categorical_param_heads.items():
            param_idx = self.categ_name_to_idx[name]
            # categorical_output: (batch, num_categ_params, embed_dim) -> (batch, embed_dim)
            param_embedding = categorical_output[:, param_idx, :]
            outputs['categorical'][name] = head(param_embedding)
        
        for name, head in self.continuius_param_heads.items():
            param_idx = self.cont_name_to_idx[name]
            # continuous_output: (batch, num_cont_params, embed_dim) -> (batch, embed_dim)
            param_embedding = continuous_output[:, param_idx, :]
            # head outputs (batch, 1), squeeze to (batch,)
            outputs['continuous'][name] = head(param_embedding).squeeze(-1)

        return outputs