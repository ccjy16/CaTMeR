import torch


def ranking_totals(scores, relevance, include):
    ranked = scores.argsort(dim=-1, descending=True)
    ordered = relevance.gather(1, ranked)[include]
    ranks = torch.arange(1, scores.shape[1] + 1, dtype=torch.float64, device=scores.device)
    ap = ((ordered.cumsum(-1) / ranks) * ordered).sum(-1) / ordered.sum(-1).clamp_min(1)
    totals = {'mAP': float(ap.sum())}
    totals.update({f'HR@{k}': int(ordered[:, :k].any(-1).sum()) for k in (1, 3, 5)})
    return totals, len(ordered)


@torch.no_grad()
def evaluate(model, loader, history, config, device):
    model.eval()
    dataset = loader.dataset
    gallery_chunks = []
    for start in range(0, len(dataset.gallery), config.valid_batch_size):
        ids = dataset.gallery[start:start + config.valid_batch_size]
        images = torch.stack([dataset.image(image_id) for image_id in ids]).to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            gallery_chunks.append(model.encode_images(images))
    gallery = torch.cat(gallery_chunks)
    image_labels = [set(dataset.evaluation_labels.get(image_id, ())) for image_id in dataset.gallery]
    totals = dict.fromkeys(('mAP', 'HR@1', 'HR@3', 'HR@5'), 0.)
    samples = evaluated = 0
    for batch in loader:
        memory = history.query(batch['scopes'], batch['positions'].tolist(), device,
                               config.history_capacity, query_speakers=batch['query_speakers'])
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            encoded = model.encode_queries(batch['texts'], batch['positions'].to(device), memory)
            scores = model.score_candidates(encoded, gallery)
        labels = batch['labels']
        relevance = torch.tensor([[int(label) in labels_for_image for labels_for_image in image_labels]
                                  for label in labels.tolist()], dtype=torch.bool, device=device)
        if config.profile == 'multichat':
            include = torch.ones(len(labels), dtype=torch.bool, device=device)
        else:
            include = batch['valid'].to(device) & labels.to(device).ge(0) & relevance.any(-1)
        batch_totals, count = ranking_totals(scores, relevance, include)
        for name, value in batch_totals.items():
            totals[name] += value
        samples += len(labels)
        evaluated += count
    return {**{key: value / max(1, evaluated) for key, value in totals.items()},
            'samples': samples, 'evaluated_queries': evaluated}
