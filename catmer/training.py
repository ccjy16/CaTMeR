import fcntl
import hashlib
import json
import logging
import math
import random
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .checkpoint import publish_epoch, read_resume
from .config import ROOT
from .data import build_memory, collate_samples, image_transform, load_dstc, load_multichat
from .evaluation import evaluate
from .model import load_model
from .objectives import training_loss, training_supervision


class MultiChatSampler(Sampler):
    def __init__(self, size, batch_size, seed):
        self.size = size
        self.padded_size = math.ceil(size / batch_size) * batch_size
        self.seed = seed
        self.epoch = 1

    def __len__(self):
        return self.padded_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch - 1)
        indices = torch.randperm(self.size, generator=generator).tolist()
        return iter((indices * math.ceil(self.padded_size / self.size))[:self.padded_size])


def build_loaders(datasets, config, device):
    loaders = {}
    for split, dataset in datasets.items():
        training = split == 'train'
        batch_size = config.batch_size if training else config.valid_batch_size
        workers = config.num_workers if training else config.valid_num_workers
        sampler = MultiChatSampler(len(dataset), batch_size, config.seed) if training and config.profile == 'multichat' else None
        loaders[split] = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                    pin_memory=device.type == 'cuda', collate_fn=collate_samples,
                                    sampler=sampler, shuffle=training and sampler is None,
                                    multiprocessing_context='spawn' if workers else None)
    return loaders


def build_optimizer(model, config):
    groups = {}
    rates = {'text': config.text_lr, 'vision': config.vision_lr, 'task': config.lr}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            family = 'text' if name.startswith('text_tower.') else 'vision' if name.startswith('visual.') else 'task'
            decay = parameter.ndim >= 2 and 'norm' not in name.lower()
            groups.setdefault((family, decay), []).append(parameter)
    return torch.optim.AdamW([
        dict(params=params, lr=rates[family], base_lr=rates[family], weight_decay=config.weight_decay if decay else 0.)
        for (family, decay), params in groups.items()
    ], betas=(.9, .999), eps=1e-8)


def scheduler_factor(config, global_step, total_steps):
    if config.profile != 'multichat':
        return 1.
    if global_step < config.warmup:
        return (global_step + 1) / config.warmup
    progress = (global_step - config.warmup) / max(1, total_steps - config.warmup)
    return .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))


def train_epoch(model, loader, history, optimizer, scaler, config, device, global_step, total_steps,
                image_labels, class_weights):
    model.train()
    totals = dict.fromkeys(('loss', 'retrieval_loss', 'auxiliary_loss', 'budget_loss'), 0.)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for step, batch in enumerate(loader, 1):
        factor = scheduler_factor(config, global_step, total_steps)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        memory = history.query(batch['scopes'], batch['positions'].tolist(), device, config.history_capacity,
                               query_speakers=batch['query_speakers'])
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            output = model(batch['texts'], batch['images'].to(device), batch['positions'].to(device), memory)
            losses = training_loss(output, batch, image_labels, class_weights, config)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(f'Nonfinite loss at step {global_step}.')
        scaler.scale(losses['loss']).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip_norm)
        if torch.isfinite(norm):
            scaler.step(optimizer)
            scaler.update()
        elif scaler.is_enabled():
            logging.warning('Skipping optimizer update at step %d due to AMP gradient overflow.', global_step)
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        else:
            raise FloatingPointError(f'Nonfinite gradient at step {global_step}.')
        global_step += 1
        for name, value in losses.items():
            totals[name] += float(value.detach())
        if step == 1 or step % 10 == 0 or step == len(loader):
            logging.info('Step %d/%d | loss %.6f | gradient norm %.4f', step, len(loader), float(losses['loss']), float(norm))
    return {name: value / len(loader) for name, value in totals.items()}, global_step


def dataset_identity(datasets, history):
    digest = hashlib.sha256()
    for split, dataset in datasets.items():
        digest.update(json.dumps([split, dataset.records, dataset.gallery, dataset.evaluation_labels],
                                 sort_keys=True, ensure_ascii=False).encode())
        if dataset.profile != 'multichat':
            for scope in sorted({record['scope'] for record in dataset.records}):
                digest.update(json.dumps([scope, dataset.image_reader.dialogues[scope]],
                                         sort_keys=True, ensure_ascii=False).encode())
    for key in history.keys:
        digest.update(json.dumps([key, history.texts[key], history.metadata[key]],
                                 sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


def train(config):
    if config.gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f'CUDA device index {config.gpu_id} is unavailable.')
    device = torch.device('cuda', config.gpu_id)
    torch.cuda.set_device(device)
    if config.resume is not None:
        root = config.resume.parent.parent
        if config.resume.parent.name != 'checkpoints' or not (root/'config.json').is_file():
            raise ValueError('Resume requires a checkpoint from a CaTMeR training run.')
    elif config.output_root is not None:
        root = config.output_root
        root.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT/'outputs'/config.profile
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
    with (root/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'A process is already training in {root}.') from error
        run_training(config, root, device)


def run_training(config, root, device):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(root/'training.log')], force=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
                if key not in {'resume', 'output_root', 'gpu_id', 'num_workers', 'valid_num_workers'}}
    if config.resume is None:
        run_config = {**settings, 'gpu_id': config.gpu_id, 'num_workers': config.num_workers,
                      'valid_num_workers': config.valid_num_workers}
        (root/'config.json').write_text(json.dumps(run_config, indent=2) + '\n', encoding='utf-8')
    logging.info('Dataset %s | device %s | run directory %s', config.profile, device, root)
    datasets = (load_multichat(config.data_root, config.multichat_annotations_root) if config.profile == 'multichat'
                else load_dstc(config.data_root, config.profile))
    transform = image_transform(config)
    for split, dataset in datasets.items():
        if not len(dataset) or not dataset.gallery:
            raise ValueError(f'The {split} split requires samples and a gallery.')
        dataset.transform = transform
        logging.info('%s | %d queries | %d gallery images', split, len(dataset), len(dataset.gallery))
    history = build_memory(datasets)
    training = datasets['train']
    class_weights, image_labels = training_supervision(training.records, training.num_classes, config.profile)
    class_weights = class_weights.to(device)
    loaders = build_loaders(datasets, config, device)
    total_steps = len(loaders['train']) * config.max_epochs
    model = load_model(config, training.num_classes)
    identity = {'data': dataset_identity(datasets, history), 'total_steps': total_steps}
    model.to(device)
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start_epoch, global_step = 1, 0
    if config.resume is not None:
        state = read_resume(config.resume, settings, identity)
        state['model'].pop('text_tower._extra_state', None)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, global_step = state['epoch'] + 1, state['global_step']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        del state
        logging.info('Resuming at epoch %d | global step %d', start_epoch, global_step)
    if start_epoch > config.max_epochs:
        logging.info('All %d epochs have already completed.', config.max_epochs)
        return
    history.refresh(model, config.history_encode_batch_size, version=global_step)
    for epoch in range(start_epoch, config.max_epochs + 1):
        if config.profile == 'multichat':
            loaders['train'].sampler.epoch = epoch
        logging.info('Epoch %d/%d | history snapshot step %d', epoch, config.max_epochs, history.version)
        losses, global_step = train_epoch(model, loaders['train'], history, optimizer, scaler, config, device,
                                          global_step, total_steps, image_labels, class_weights)
        history.refresh(model, config.history_encode_batch_size, version=global_step)
        metrics = {}
        for split in ('val', 'test'):
            values = evaluate(model, loaders[split], history, config, device)
            metrics[split] = values
            logging.info('%s | mAP %.4f%% | HR@1 %.4f%% | HR@3 %.4f%% | HR@5 %.4f%% | queries %d/%d',
                         split, *(100. * values[key] for key in ('mAP', 'HR@1', 'HR@3', 'HR@5')),
                         values['evaluated_queries'], values['samples'])
        record = dict(epoch=epoch, global_step=global_step, train=losses, **metrics)
        state = dict(epoch=epoch, global_step=global_step, config=settings, identity=identity, epoch_metrics=record,
                     model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        logging.info('Saved %s', publish_epoch(root, state))
        del state
