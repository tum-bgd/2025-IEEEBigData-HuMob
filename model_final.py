import torch
from torch import nn


class DayEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(DayEmbeddingModel, self).__init__()

        self.day_embedding = nn.Embedding(
            num_embeddings=75+1,
            embedding_dim=embed_size,
        )

    def forward(self, day):
        embed = self.day_embedding(day)
        return embed


class TimeEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(TimeEmbeddingModel, self).__init__()

        self.time_embedding = nn.Embedding(
            num_embeddings=48+1,
            embedding_dim=embed_size,
        )

    def forward(self, time):
        embed = self.time_embedding(time)
        return embed


class LocationXEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    201: <MASK>
    '''
    def __init__(self, embed_size):
        super(LocationXEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=202,
            embedding_dim=embed_size,
        )

    def forward(self, location):
        embed = self.location_embedding(location)
        return embed


class LocationYEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    201: <MASK>
    '''
    def __init__(self, embed_size):
        super(LocationYEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=202,
            embedding_dim=embed_size,
        )

    def forward(self, location):
        embed = self.location_embedding(location)
        return embed


class TimedeltaEmbeddingModel(nn.Module):
    '''
    0: <PAD>
    '''
    def __init__(self, embed_size):
        super(TimedeltaEmbeddingModel, self).__init__()

        self.timedelta_embedding = nn.Embedding(
            num_embeddings=48,
            embedding_dim=embed_size,
        )

    def forward(self, timedelta):
        embed = self.timedelta_embedding(timedelta)
        return embed


class WeeklyPosEmbedding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.weekly_pos_embed = nn.Embedding(8, embed_size)

    def forward(self, day):
        idx = (day % 7) + 1
        idx = torch.where(day == 0,
                          torch.zeros_like(idx),
                          idx)
        return self.weekly_pos_embed(idx)


class FeatureWiseInteraction(nn.Module):
    def __init__(self, embed_size: int, nhead: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_size, nhead,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stack):
        B, L, F, E = stack.shape
        x = stack.view(B * L, F, E)
        out, _ = self.attn(x, x, x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return out.view(B, L, E)


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size: int, fwi_heads: int = 2):
        super().__init__()
        self.day_emb       = DayEmbeddingModel(embed_size)
        self.time_emb      = TimeEmbeddingModel(embed_size)
        self.loc_x_emb     = LocationXEmbeddingModel(embed_size)
        self.loc_y_emb     = LocationYEmbeddingModel(embed_size)
        self.timedelta_emb = TimedeltaEmbeddingModel(embed_size)
        self.weekly_emb    = WeeklyPosEmbedding(embed_size)
        self.DayorNight_emb    = DayorNightEmbedding(embed_size)
        self.POI_emb    = POIEmbedding(embed_size)
        self.fwi = FeatureWiseInteraction(embed_size,
                                          nhead=fwi_heads,
                                          dropout=0.1)

    def forward(self, day, time, loc_x, loc_y, timedelta_):
        e_day  = self.day_emb(day)
        e_time = self.time_emb(time)
        e_x    = self.loc_x_emb(loc_x)
        e_y    = self.loc_y_emb(loc_y)
        e_td   = self.timedelta_emb(timedelta_)
        e_w    = self.weekly_emb(day)
        e_poi    = self.POI_emb(day)
        e_dn    = self.DayorNight_emb(day)

        stack = torch.stack([e_day, e_time, e_x, e_y, e_td, e_w,e_poi,e_dn], dim=2)
        fused = stack.sum(dim=2)
        inter = self.fwi(stack)
        return fused + inter

# ===========================================================
# 4. Transformer Encoder
# ===========================================================
class TransformerEncoderModel(nn.Module):
    def __init__(self, layers_num: int, heads_num: int, embed_size: int,
                 dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads_num,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers_num)

    def forward(self, x, src_key_padding_mask):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class DeeperFFNLayer(nn.Module):
    def __init__(self, embed_size: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        def _branch():
            return nn.Sequential(
                nn.Linear(embed_size, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 200)
            )
        self.ffn_x = _branch()
        self.ffn_y = _branch()

    def forward(self, x):
        out_x = self.ffn_x(x)
        out_y = self.ffn_y(x)
        return torch.stack([out_x, out_y], dim=2)


class DistanceFFNLayer(nn.Module):
    def __init__(self, embed_size, hidden_dim=128, dropout=0.12):
        super().__init__()
        self.ffn_x = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )
        self.ffn_y = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x):
        out_x = self.ffn_x(x)
        out_y = self.ffn_y(x)
        return torch.stack([out_x, out_y], dim=2)

class DirectionFFNLayer(nn.Module):
    def __init__(self, embed_size, hidden_dim=128, dropout=0.12):
        super().__init__()
        self.ffn_x = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.ffn_y = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        out_x = self.ffn_x(x)
        out_y = self.ffn_y(x)
        return torch.stack([out_x, out_y], dim=2)


class HumobTransformer(nn.Module):
    def __init__(self,
                 layers_num: int,
                 heads_num: int,
                 embed_size: int,
                 use_subtask1: bool = False,
                 use_subtask2: bool = False):
        super(HumobTransformer, self).__init__()
        self.use_subtask1 = use_subtask1
        self.use_subtask2 = use_subtask2

        self.embedding_layer = EmbeddingLayer(embed_size, fwi_heads=8)
        self.transformer_encoder = TransformerEncoderModel(
            layers_num, heads_num, embed_size)
        self.ffn_layer_main = DeeperFFNLayer(embed_size)
        if self.use_subtask1:
            self.ffn_layer_distance = DistanceFFNLayer(embed_size)
        if self.use_subtask2:
            self.ffn_layer_direction = DirectionFFNLayer(embed_size)

    def forward(self,
                day, time, location_x, location_y, timedelta_,
                seq_lens):
        embed = self.embedding_layer(
            day, time, location_x, location_y, timedelta_)
        B, L = day.shape
        idxs = torch.arange(L, device=seq_lens.device).expand(B, L)
        src_key_padding_mask = idxs >= seq_lens.unsqueeze(1)
        enc_out = self.transformer_encoder(
            embed, src_key_padding_mask=src_key_padding_mask)
        outputs = {}
        outputs["main_out"] = self.ffn_layer_main(enc_out)

        if self.use_subtask1:
            outputs["distance_out"] = self.ffn_layer_distance(enc_out)

        if self.use_subtask2:
            outputs["direction_out"] = self.ffn_layer_direction(enc_out)

        return outputs

