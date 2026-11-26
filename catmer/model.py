import math

import torch
from torch import nn
from torch.nn import functional as F

from .memory import CandidateTextMemoryReader, SparseHistorySelector
from .text import TrainableGraniteTextTower
from .vision import PVTMultiGranularityVisualEncoder, load_pvt_backbone


MATCHING_INITIALIZATION_DRAWS = 985_220


class CaTMeR(nn.Module):
    def __init__(self, text_tower, visual, num_classes, config):
        super().__init__()
        self.text_tower, self.visual = text_tower, visual
        dimension = config.retrieval_dim
        self.text_projection = nn.Linear(text_tower.output_dim, dimension)
        torch.rand(MATCHING_INITIALIZATION_DRAWS, device='cpu')
        self.selector = SparseHistorySelector(dimension, config.semantic_scale_initial)
        self.classifier = nn.Linear(2 * dimension, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / .07)))
        self.fusion = nn.Linear(3 * dimension, dimension)
        self.candidate_reader = CandidateTextMemoryReader(config)
        self.distance_bias = nn.Embedding(config.distance_buckets, 1)
        nn.init.zeros_(self.distance_bias.weight)

    def encode_history(self, texts):
        return F.normalize(self.text_projection(self.text_tower(texts)).float(), dim=-1)

    def encode_queries(self, texts, positions, history):
        q = self.encode_history(texts)
        if history is None:
            features = q.new_empty(len(q), 0, q.shape[-1])
            history_positions = positions.new_empty(len(q), 0)
            mask = torch.zeros((len(q), 0), dtype=torch.bool, device=q.device)
            metadata = {'distance_known': mask}
        else:
            features, history_positions, mask, metadata = history
        memory, weights, penalty = self.selector(q, features, positions, history_positions, mask,
                                                  metadata['distance_known'], self.distance_bias)
        auxiliary = self.classifier(torch.cat((q, memory), dim=-1))
        fused = F.normalize(self.fusion(torch.cat((q, q, memory), dim=-1)).float(), dim=-1)
        return dict(query_features=fused, auxiliary_logits=auxiliary, query=q, history=features,
                    history_mask=mask, history_weights=weights, budget_loss=penalty,
                    positions=positions, history_positions=history_positions, metadata=metadata)

    def encode_images(self, images):
        return F.normalize(self.visual(images).float(), dim=-1)

    def score_candidates(self, encoded, image_features):
        with torch.autocast(device_type=image_features.device.type, enabled=False):
            base = encoded['query_features'].float() @ image_features.float().T
        delta = self.candidate_reader(encoded['query'], image_features, encoded['history'], encoded['history_mask'],
                                     encoded['history_weights'], encoded['positions'], encoded['history_positions'],
                                     encoded['metadata'])
        return base + delta

    def forward(self, texts, images, positions, history):
        encoded = self.encode_queries(texts, positions, history)
        scores = self.score_candidates(encoded, self.encode_images(images))
        return dict(logits=self.logit_scale.exp().clamp(max=100.) * scores,
                    auxiliary_logits=encoded['auxiliary_logits'], budget_loss=encoded['budget_loss'])


def load_model(config, num_classes):
    text = TrainableGraniteTextTower.from_pretrained(config.granite_text_model_path, config.context_length)
    visual = PVTMultiGranularityVisualEncoder(config.resolution, config.pvt_local_k, config.retrieval_dim)
    load_pvt_backbone(visual, config.pvt_checkpoint)
    model = CaTMeR(text, visual, num_classes, config)
    text.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model
