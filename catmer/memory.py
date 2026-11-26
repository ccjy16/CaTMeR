import math

import torch
from torch import nn
from torch.nn import functional as F


def relative_distance_buckets(query_positions, history_positions, valid_mask, distance_known, num_buckets):
    delta = query_positions.to(torch.float64).unsqueeze(-1) - history_positions.to(torch.float64)
    known = valid_mask & distance_known & torch.isfinite(delta) & delta.ge(0)
    safe = torch.where(known, delta, 0.)
    positive = (torch.floor(torch.log2(safe.clamp_min(1))) + 2).long().clamp(max=num_buckets - 1)
    return torch.where(safe.eq(0), 1, positive).masked_fill(~known, 0)


def sparse_memory_weights(scores, mask, thresholds):
    safe_scores = scores.masked_fill(~mask, 0.)
    soft = torch.sigmoid(safe_scores - thresholds)
    sparse = (2. * soft - 1.).clamp_min(0.)
    gates = (soft + (sparse - soft).detach()).masked_fill(~mask, 0.)
    best = safe_scores.masked_fill(~mask, -torch.inf).max(-1, keepdim=True).values
    best = torch.where(mask.any(-1, keepdim=True), best, 0.)
    rank = torch.exp((safe_scores - best).masked_fill(~mask, -torch.inf))
    contributions = gates * rank
    return contributions / contributions.sum(-1, keepdim=True).clamp_min(1.)


class SparseHistorySelector(nn.Module):
    def __init__(self, dimension, semantic_scale):
        super().__init__()
        raw_scale = math.log(math.expm1(semantic_scale)) if semantic_scale < 20 else semantic_scale
        self.semantic_scale_raw = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))
        self.query_proj = nn.Linear(dimension, dimension)
        self.key_proj = nn.Linear(dimension, dimension)
        self.value_proj = nn.Linear(dimension, dimension)
        self.embed_dim_mlp = nn.Sequential(nn.Linear(dimension, dimension))
        self.threshold_base = nn.Parameter(torch.zeros(()))
        self.threshold_norm = nn.LayerNorm(dimension)
        hidden = max(dimension // 4, 32)
        self.threshold_net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.threshold_net[-1].weight)
        nn.init.zeros_(self.threshold_net[-1].bias)

    def thresholds(self, query_projection):
        return self.threshold_base + self.threshold_net(self.threshold_norm(query_projection))

    def forward(self, query, history, query_positions, positions, mask, distance_known, distance_bias):
        if mask.any():
            buckets = relative_distance_buckets(query_positions, positions, mask, distance_known,
                                               distance_bias.num_embeddings)
            projected_query = self.query_proj(query)
            safe = history.masked_fill(~mask.unsqueeze(-1), 0.)
            keys, values = self.key_proj(safe), self.value_proj(safe)
            semantic = (projected_query.unsqueeze(1) * keys).sum(-1) / math.sqrt(history.shape[-1])
            scores = F.softplus(self.semantic_scale_raw) * semantic + distance_bias(buckets).squeeze(-1)
            weights = sparse_memory_weights(scores, mask, self.thresholds(projected_query))
            embedded = F.normalize(self.embed_dim_mlp(values), dim=-1, eps=1e-6)
            memory = (weights.unsqueeze(-1) * embedded).sum(1)
        else:
            memory, scores, weights = torch.zeros_like(query), history.new_zeros(mask.shape), history.new_zeros(mask.shape)
        with torch.autocast(device_type=query.device.type, enabled=False):
            thresholds = self.thresholds(self.query_proj(query.float()))
            soft = torch.sigmoid(scores.float().masked_fill(~mask, 0.) - thresholds).masked_fill(~mask, 0.)
            count = mask.sum(-1)
            fraction = soft.sum(-1) / count.clamp_min(1)
            zero = self.threshold_base * 0. + query.sum() * 0.
            penalty = fraction.sum() / count.gt(0).sum().clamp_min(1) + zero
        return memory, weights, penalty


class RelativeHistoryEncoder(nn.Module):
    def __init__(self, dimension, heads, strength, num_buckets):
        super().__init__()
        self.heads = heads
        self.strength = strength
        self.num_buckets = num_buckets
        self.distance = nn.Embedding(num_buckets, dimension)
        self.source = nn.Embedding(3, dimension)
        self.speaker = nn.Embedding(3, dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=False)
        self.order_output = nn.Linear(dimension, dimension, bias=False)
        self.relative_bias = nn.Embedding(num_buckets, heads)
        nn.init.zeros_(self.relative_bias.weight)
        for embedding in (self.distance, self.source, self.speaker):
            nn.init.normal_(embedding.weight, std=.02)

    def forward(self, history, mask, query_positions, positions, metadata):
        known = metadata['distance_known'] & mask & torch.isfinite(positions)
        buckets = relative_distance_buckets(query_positions, positions, mask, known, self.num_buckets)
        known = known & buckets.gt(0)
        features = history.masked_fill(~mask.unsqueeze(-1), 0.)
        features = features + self.strength * (self.distance(buckets) + self.source(metadata['source_types'])
                                               + self.speaker(metadata['speaker_relations']))
        features = features.masked_fill(~mask.unsqueeze(-1), 0.)
        batch, count, width = features.shape
        head_width = width // self.heads
        q, k, v = self.qkv(features).reshape(batch, count, 3, self.heads, head_width).unbind(2)
        scores = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(head_width)
        pair_known = known[:, :, None] & known[:, None, :]
        delta = positions.to(torch.float64)[:, :, None] - positions.to(torch.float64)[:, None, :]
        self_edge = torch.eye(count, dtype=torch.bool, device=history.device)[None]
        allowed = mask[:, :, None] & mask[:, None, :] & ((pair_known & delta.ge(0)) | self_edge)
        pair_buckets = relative_distance_buckets(
            positions.reshape(-1), positions[:, None, :].expand(-1, count, -1).reshape(-1, count),
            allowed.reshape(-1, count), pair_known.reshape(-1, count), self.num_buckets,
        ).reshape(batch, count, count)
        scores = scores + self.relative_bias(pair_buckets).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        probability = torch.softmax(scores, -1) * allowed[:, None]
        probability = probability / probability.sum(-1, keepdim=True).clamp_min(1.)
        ordered = torch.einsum('bhij,bjhd->bihd', probability, v).reshape(batch, count, width)
        features = features + self.strength * self.order_output(ordered)
        return F.normalize(features, dim=-1, eps=1e-6).masked_fill(~mask.unsqueeze(-1), 0.)


class CandidateTextMemoryReader(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension, hidden = config.retrieval_dim, config.candidate_hidden_dim
        self.alpha = config.candidate_alpha
        self.temperature = config.candidate_temperature
        self.chunk_size = config.candidate_chunk_size
        self.query_projection = nn.Linear(dimension, hidden, bias=False)
        self.candidate_projection = nn.Linear(dimension, hidden, bias=False)
        self.history_projection = nn.Linear(dimension, hidden, bias=False)
        self.score = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)
        self.temporal_encoder = RelativeHistoryEncoder(hidden, config.history_heads, config.history_strength,
                                                       config.distance_buckets)

    def forward(self, query, candidates, history, mask, weights, query_positions, positions, metadata):
        if not history.shape[1] or not len(candidates):
            return query.new_zeros(len(query), len(candidates))
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = F.normalize(self.query_projection(query.float()), dim=-1, eps=1e-6)
            c = F.normalize(self.candidate_projection(candidates.float()), dim=-1, eps=1e-6)
            safe = history.float().masked_fill(~mask.unsqueeze(-1), 0.)
            h = F.normalize(self.history_projection(safe), dim=-1, eps=1e-6)
            weights = weights.float().masked_fill(~mask, 0.)
            if not torch.isfinite(weights).all() or weights.lt(0).any():
                raise ValueError('History weights must be finite and nonnegative.')
            support = mask & weights.gt(0)
            h = self.temporal_encoder(h, support, query_positions, positions, metadata)
            mass = weights.sum(-1, keepdim=True).clamp(max=1.)
            deltas = []
            for chunk in c.split(self.chunk_size):
                joint = q[:, None, :] + chunk[None, :, :]
                scores = torch.einsum('bld,bhd->blh', joint, h) / self.temperature
                scores = scores.masked_fill(~support[:, None, :], torch.finfo(scores.dtype).min)
                unnormalized = torch.exp(scores - scores.amax(-1, keepdim=True)) * weights[:, None, :]
                attention = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
                context = torch.einsum('blh,bhd->bld', attention, h)
                features = torch.cat((q[:, None, :] * context, chunk[None, :, :] * context, context), -1)
                deltas.append(self.alpha * torch.tanh(self.score(features).squeeze(-1)) * mass)
        return torch.cat(deltas, dim=1)
