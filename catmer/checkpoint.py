import json
import os
from pathlib import Path
import tempfile

import torch


def epoch_path(root, epoch):
    return root/'checkpoints'/f'epoch_{epoch:04d}.pt'


def _write_atomic(path, write, mode):
    options = {} if mode == 'wb' else {'encoding': 'utf-8'}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **options) as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_metrics(root, records):
    content = ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n' for record in records)
    _write_atomic(root/'metrics.jsonl', lambda handle: handle.write(content), 'w')


def _read_metrics(root):
    path = root/'metrics.jsonl'
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return {record['epoch']: record for record in records}


def publish_epoch(root, state):
    epoch = state['epoch']
    path = epoch_path(root, epoch)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Epoch checkpoint already exists: {path}')
    records = _read_metrics(root)
    if set(records) != set(range(1, epoch)):
        raise RuntimeError('Previous metrics do not match the completed epochs. Resume the run to reconcile them.')
    _write_atomic(path, lambda handle: torch.save(state, handle), 'wb')
    _write_metrics(root, [*records.values(), state['epoch_metrics']])
    return path


def read_resume(path, expected_config=None, expected_identity=None):
    root = path.parent.parent
    completed = sorted(path.parent.glob('epoch_*.pt'), key=lambda item: int(item.stem.removeprefix('epoch_')))
    if path.parent.name != 'checkpoints' or not completed or completed[-1] != path:
        raise ValueError('Resume requires the most recent completed epoch in the run checkpoint directory.')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if expected_config is not None and state['config'] != expected_config:
        changed = [key for key in expected_config if state['config'].get(key) != expected_config[key]]
        raise ValueError(f'The resumed run requires identical training settings. Changed: {changed}')
    if expected_identity is not None and any(state['identity'].get(key) != value
                                             for key, value in expected_identity.items()):
        raise ValueError('The dataset or planned training steps changed since this run began.')
    epoch = state['epoch']
    if completed != [epoch_path(root, number) for number in range(1, epoch + 1)]:
        raise ValueError('Checkpoint names do not match a contiguous epoch sequence.')
    try:
        records = _read_metrics(root)
    except (ValueError, KeyError, TypeError):
        records = {}
    records[epoch] = state['epoch_metrics']
    for number in range(1, epoch):
        if number not in records:
            previous = torch.load(epoch_path(root, number), map_location='cpu', weights_only=False)
            records[number] = previous['epoch_metrics']
            del previous
    _write_metrics(root, [records[number] for number in range(1, epoch + 1)])
    return state
