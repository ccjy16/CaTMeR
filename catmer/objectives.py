from collections import defaultdict

import torch
from torch.nn import functional as F


def training_supervision(records, num_classes, profile):
    counts = [0] * num_classes
    image_labels = defaultdict(set)
    for record in records:
        label = record['label']
        if record['valid'] and 0 <= label < num_classes:
            counts[label] += 1
            if profile == 'multichat':
                image_labels[record['image_id']].add(label)
    counts = torch.tensor(counts, dtype=torch.float32)
    observed = counts.gt(0)
    weights = torch.where(observed, counts.clamp_min(1).rsqrt(), 0.)
    if observed.any():
        weights = weights / weights[observed].mean()
    return weights, image_labels


def semantic_targets(labels, image_ids, image_labels, selected, profile):
    if profile != 'multichat':
        known = selected[:, None] & selected[None, :]
        return labels[:, None].eq(labels[None, :]) & known, known
    candidate_labels = [image_labels.get(image_id, ()) for image_id in image_ids]
    candidate_known = torch.tensor([bool(values) for values in candidate_labels], device=labels.device)
    known = selected[:, None] & candidate_known[None, :]
    positives = torch.tensor([[label in values for values in candidate_labels] for label in labels.tolist()],
                             dtype=torch.bool, device=labels.device)
    return positives & known, known


def balanced_centered_bce(logits, positive, known):
    negative = known & ~positive
    positives, negatives = positive.sum(-1), negative.sum(-1)
    scores = torch.where(known, logits.float(), 0.)
    if scores.shape[1]:
        reference = scores.masked_fill(~known, -torch.inf).amax(-1, keepdim=True)
        reference = torch.where(known.any(-1, keepdim=True), reference, 0.).detach()
        scores = torch.where(known, scores - reference, 0.)
    center = scores.sum(-1, keepdim=True) / known.sum(-1, keepdim=True).clamp_min(1)
    scores = torch.where(known, scores - center, 0.)
    eligible = positives.gt(0) & negatives.gt(0)
    positive_cost = torch.where(positive, F.softplus(-scores), 0.).sum(-1) / positives.clamp_min(1)
    negative_cost = torch.where(negative, F.softplus(scores), 0.).sum(-1) / negatives.clamp_min(1)
    return torch.where(eligible, .5 * (positive_cost + negative_cost), 0.).sum() / eligible.sum().clamp_min(1)


def training_loss(output, batch, image_labels, class_weights, config):
    logits = output['logits']
    labels = batch['labels'].to(logits.device)
    selected = labels.ge(0) & labels.lt(len(class_weights)) & batch['valid'].to(logits.device)
    positive, known = semantic_targets(labels, batch['image_ids'], image_labels, selected, config.profile)
    if not torch.isfinite(logits[known]).all():
        raise FloatingPointError('Nonfinite logits for supervised retrieval pairs.')
    retrieval = balanced_centered_bce(logits, positive, known)
    selected = selected & class_weights[labels.clamp(0, len(class_weights) - 1)].gt(0)
    auxiliary_logits = output['auxiliary_logits']
    auxiliary = (F.cross_entropy(auxiliary_logits[selected].float(), labels[selected], weight=class_weights)
                 if selected.any() else auxiliary_logits.reshape(-1)[:0].sum())
    budget = output['budget_loss']
    total = .5 * (retrieval + auxiliary) + config.budget_weight * budget
    return dict(loss=total, retrieval_loss=retrieval, auxiliary_loss=auxiliary, budget_loss=budget)
