import ast
import base64
import bisect
import json
import pickle
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import lmdb
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
MODES = ('boba', 'kuaile', 'quanguo', 'siban', 'yongyuan')


def load_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

class StickerDataset(Dataset):
    def __init__(self, records, gallery, image_reader, *, split, profile, num_classes,
                 evaluation_labels, transform=None, memory_texts=None, memory_metadata=None):
        self.records, self.gallery, self.image_reader = records, list(gallery), image_reader
        self.split, self.profile, self.num_classes = split, profile, num_classes
        self.evaluation_labels, self.transform = evaluation_labels, transform
        self.memory_texts = memory_texts or {}
        self.memory_metadata = memory_metadata or {}

    def __len__(self):
        return len(self.records)

    def query_text(self, record):
        if 'query_text' in record:
            return record['query_text']
        turns = self.image_reader.dialogues[record['scope']]
        return '[SEP]'.join(f"{t['speaker_id']}: {t.get('txt', '')}".rstrip()
                           for t in turns[:record['position'] + 1])

    def image(self, image_id):
        image = self.image_reader.read(str(image_id))
        return self.transform(image) if self.transform is not None else image

    def __getitem__(self, index):
        record = dict(self.records[index % len(self.records)])
        record.update(text=self.query_text(record), image=self.image(record['image_id']),
                      query_speaker=self.query_speaker(record))
        return record

    def query_speaker(self, record):
        if self.profile == 'multichat':
            return record.get('speaker')
        turn = self.image_reader.dialogues[record['scope']][record['position']]
        return turn.get('speaker_id')

class LMDBImages:
    def __init__(self, path):
        self.path, self.env = str(path), None

    def read(self, image_id):
        if self.env is None:
            self.env = lmdb.open(self.path, readonly=True, create=False, lock=False, readahead=False)
        with self.env.begin() as txn:
            raw = txn.get(image_id.encode())
        if raw is None:
            raise KeyError(f'Image {image_id} is absent from {self.path}')
        with Image.open(BytesIO(base64.urlsafe_b64decode(raw))) as image:
            image.load()
            return image.convert('RGB')

    def __getstate__(self):
        return {'path': self.path, 'env': None}

class DSTCImages:
    def __init__(self, root, dialogues):
        self.root, self.dialogues = Path(root), dialogues

    def read(self, image_id):
        with Image.open(self.root / 'images' / f'{image_id}.png') as image:
            image.load()
            return image.convert('RGB')

def load_multichat(root, annotations):
    mapping = {}
    for line in (annotations / 'all' / 'intent.txt').read_text().splitlines():
        fields = line.split('\t', 1)
        if len(fields) == 2:
            mapping.setdefault(fields[1].strip().lower(), int(fields[0]))
    labels = defaultdict(set)
    for source_mode in MODES:
        path = annotations / f'{source_mode}_sample_intent_style_attribute.json'
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = ast.literal_eval(line)
                labels[str(item['image_ids'])].add(mapping[str(item['fined_intent']).strip().lower()])
    datasets = {}
    for split in ('train', 'val', 'test'):
        records, memories = [], {}
        env = lmdb.open(str(Path(root) / split / 'pairs'), readonly=True, create=False, lock=False)
        with env.begin() as txn:
            count = int(txn.get(b'num_samples'))
            for index in range(count):
                pair = pickle.loads(txn.get(str(index).encode()))
                if len(pair) != 10:
                    raise ValueError('Original all records must include source_mode')
                sid, sample, summary, _, context, speaker, _, intent, image_id, source_mode = pair
                if source_mode not in MODES:
                    raise ValueError(f'Unknown mode {source_mode}')
                record = dict(scope=str(source_mode), position=int(sid), sample_id=str(sample),
                    image_id=str(image_id), label=mapping[str(intent).strip().lower()], valid=True,
                    query_text='[SEP]'.join(str(context).split('\t')).lower(), summary=str(summary),
                    speaker=str(speaker))
                records.append(record)
                key = (record['scope'], record['position'])
                if key in memories and memories[key] != str(summary):
                    raise ValueError(f'Conflicting summaries for session {key}')
                memories[key] = str(summary)
        env.close()
        image_path = Path(root) / f'{split}_imgs'
        env = lmdb.open(str(image_path), readonly=True, create=False, lock=False)
        with env.begin() as txn:
            gallery = sorted(k.decode() for k, _ in txn.cursor() if k != b'num_images')
        env.close()
        gallery_ids = set(gallery)
        if any(r['image_id'] not in gallery_ids for r in records):
            raise ValueError(f'Missing targets in {split} gallery')
        datasets[split] = StickerDataset(records, gallery, LMDBImages(image_path), split=split,
            profile='multichat', num_classes=max(mapping.values()) + 1,
            evaluation_labels={k: sorted(v) for k, v in labels.items()}, memory_texts=memories,
            memory_metadata={key: dict(source_type=1, speaker=None) for key in memories})
    return datasets

def load_dstc(root, profile):
    root = Path(root)
    dialogues = {str(d['dialogue_id']): d['turns'] for d in load_jsonl(root / 'dialogues.jsonl')}
    gallery = sorted(p.stem for p in (root / 'images').glob('*.png'))
    labels = json.loads((root / 'labels' / 'sticker_emotion_top2_train.json').read_text())
    reader = DSTCImages(root, dialogues)
    datasets = {}
    for split in ('train', 'val', 'test'):
        records, memories, memory_metadata = [], {}, {}
        for source in load_jsonl(root / 'splits' / f'{split}.jsonl'):
            scope, position = str(source['dialogue_id']), int(source['turn_id'])
            if position < 0 or position >= len(dialogues[scope]):
                raise ValueError('Target turn is outside its dialogue')
            records.append(dict(scope=scope, position=position, sample_id=str(source['sample_id']),
                image_id=str(source['target_sticker_id']), label=int(source.get('emotion_id', -1)),
                valid=bool(source.get('emotion_valid', False))))
        for scope in {r['scope'] for r in records}:
            for position, turn in enumerate(dialogues[scope]):
                memories[(scope, position)] = f"{turn['speaker_id']}: {turn.get('txt', '')}".rstrip()
                memory_metadata[(scope, position)] = dict(source_type=2, speaker=turn.get('speaker_id'))
        gallery_ids = set(gallery)
        if any(r['image_id'] not in gallery_ids for r in records):
            raise ValueError(f'Missing targets in {split} gallery')
        datasets[split] = StickerDataset(records, gallery, reader, split=split, profile=profile,
            num_classes=52, evaluation_labels=labels, memory_texts=memories, memory_metadata=memory_metadata)
    return datasets

def image_transform(args):
    interpolation = InterpolationMode.BICUBIC if args.profile == 'multichat' else InterpolationMode.BILINEAR
    return transforms.Compose([transforms.Resize((args.resolution, args.resolution), interpolation),
        transforms.ToTensor(), transforms.Normalize((.48145466, .4578275, .40821073),
                                                   (.26862954, .26130258, .27577711))])

def collate_samples(items):
    return dict(texts=[r['text'] for r in items], images=torch.stack([r['image'] for r in items]),
        scopes=[r['scope'] for r in items], positions=torch.tensor([r['position'] for r in items]),
        labels=torch.tensor([r['label'] for r in items]), valid=torch.tensor([r['valid'] for r in items]),
        image_ids=[r['image_id'] for r in items], sample_ids=[r['sample_id'] for r in items],
        query_speakers=[r.get('query_speaker', r.get('speaker')) for r in items])

class HistorySnapshot:
    def __init__(self, texts, metadata=None):
        self.texts = dict(texts)
        self.metadata = dict(metadata or {})
        self.keys = sorted(texts)
        self.features, self.version = None, None
        self.scope_rows = defaultdict(list)
        for row, (scope, position) in enumerate(self.keys):
            self.scope_rows[scope].append((position, row))
        self.scope_rows = {scope: (tuple(p for p, _ in rows), tuple(r for _, r in rows))
                           for scope, rows in self.scope_rows.items()}

    def refresh(self, model, batch_size=64, version=None):
        chunks, was_training = [], model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in tqdm(range(0, len(self.keys), batch_size), desc='History refresh',
                                  unit='batch', mininterval=2., disable=len(self.keys) < 100):
                    features = model.encode_history([self.texts[k] for k in self.keys[start:start+batch_size]])
                    chunks.append(features.detach().cpu().float())
        finally:
            model.train(was_training)
        features = torch.cat(chunks) if chunks else torch.empty(0, 512)
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite history snapshot; previous version retained')
        self.features, self.version = features, version

    def selected_rows(self, scope, position, limit):
        positions, rows = self.scope_rows.get(scope, ((), ()))
        end = bisect.bisect_left(positions, int(position))
        return positions[max(0, end-limit):end], rows[max(0, end-limit):end]

    def query(self, scopes, positions, device, limit, *, query_speakers):
        if self.features is None:
            raise RuntimeError('Refresh the history snapshot before querying it')
        selected = [self.selected_rows(scope, position, limit) for scope, position in zip(scopes, positions)]
        width = max((len(rows) for _, rows in selected), default=0)
        if width == 0:
            return None
        features = torch.zeros(len(scopes), width, 512)
        ids = torch.zeros(len(scopes), width, dtype=torch.long)
        mask = torch.zeros(len(scopes), width, dtype=torch.bool)
        sources, relations = torch.zeros_like(ids), torch.zeros_like(ids)
        for i, (local_positions, rows) in enumerate(selected):
            count = len(rows)
            if count:
                features[i, :count] = self.features[list(rows)]
                ids[i, :count] = torch.tensor(local_positions)
                mask[i, :count] = True
                current_speaker = query_speakers[i]
                for column, row in enumerate(rows):
                    item = self.metadata.get(self.keys[row], {})
                    source = int(item.get('source_type', 0))
                    sources[i, column] = source
                    speaker = item.get('speaker')

                    if source == 2 and speaker is not None and current_speaker is not None and str(speaker) and str(current_speaker):
                        relations[i, column] = 1 if str(speaker) == str(current_speaker) else 2
        result = tuple(value.to(device) for value in (features, ids, mask))
        return (*result, dict(source_types=sources.to(device), speaker_relations=relations.to(device),
                              distance_known=result[2]))

def build_memory(datasets):
    texts, metadata = {}, {}
    for dataset in datasets.values():
        for key, text in dataset.memory_texts.items():
            if key in texts and texts[key] != text:
                raise ValueError(f'Conflicting history text across splits: {key}')
            texts[key] = text
            item = getattr(dataset, 'memory_metadata', {}).get(key, {})
            if key in metadata and metadata[key] != item:
                raise ValueError(f'Conflicting history provenance across splits: {key}')
            metadata[key] = item
    return HistorySnapshot(texts, metadata)
