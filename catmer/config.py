import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT.parent
PROFILES = {
    'multichat': ('MultiChat/lmdb_all_intent_style_attribute', 123, 30, 100, 8),
    'dstc-zh': ('DSTC10-MOD/DSTC10-MOD-MGTM_IGSR', 42, 10, 0, 64),
    'dstc-en': ('DSTC10-MOD/DSTC10-MOD-MGTM_IGSR_EN', 42, 10, 0, 64),
}


@dataclass(frozen=True)
class TrainingConfig:
    profile: str
    data_root: Path
    multichat_annotations_root: Path | None
    granite_text_model_path: Path
    pvt_checkpoint: Path
    output_root: Path | None
    resume: Path | None
    gpu_id: int
    max_epochs: int
    batch_size: int
    valid_batch_size: int
    num_workers: int
    valid_num_workers: int
    seed: int
    warmup: int
    history_encode_batch_size: int
    history_capacity: int = 256
    pvt_local_k: int = 16
    distance_buckets: int = 18
    context_length: int = 512
    resolution: int = 224
    retrieval_dim: int = 512
    candidate_hidden_dim: int = 128
    candidate_alpha: float = .1
    candidate_temperature: float = .2
    candidate_chunk_size: int = 64
    history_heads: int = 4
    history_strength: float = .1
    semantic_scale_initial: float = 100.
    budget_weight: float = .001
    text_lr: float = 5e-6
    vision_lr: float = 1e-5
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train CaTMeR for conversational sticker retrieval.')
    parser.add_argument('profile', choices=PROFILES)
    parser.add_argument('--data-root', type=Path, help='Preprocessed dataset directory')
    parser.add_argument('--multichat-annotations-root', type=Path, help='MultiChat directory containing all/intent.txt and the five mode annotations')
    parser.add_argument('--granite-text-model-path', type=Path, default=ASSET_ROOT/'pretrained_weights/granite-embedding-311m-multilingual-r2')
    parser.add_argument('--pvt-checkpoint', type=Path, default=ASSET_ROOT/'pretrained_weights/PVT_v2_b2.pth')
    parser.add_argument('--output-root', type=Path, help='Directory for a new training run')
    parser.add_argument('--resume', type=Path, help='Latest completed epoch checkpoint of the same run')
    parser.add_argument('--gpu-id', type=int, default=0, help='Index within CUDA_VISIBLE_DEVICES (default: 0)')
    parser.add_argument('--max-epochs', type=int, help='Default: 30 for MultiChat and 10 for DSTC10-MOD')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--valid-batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--valid-num-workers', type=int, default=0)
    args = parser.parse_args(argv)
    dataset, seed, epochs, warmup, history_batch = PROFILES[args.profile]
    if args.max_epochs is None:
        args.max_epochs = epochs
    if min(args.max_epochs, args.batch_size, args.valid_batch_size) < 1:
        parser.error('Epoch and batch counts must be positive.')
    if min(args.gpu_id, args.num_workers, args.valid_num_workers) < 0:
        parser.error('GPU index and worker counts must be nonnegative.')
    if args.profile == 'multichat':
        args.multichat_annotations_root = args.multichat_annotations_root or ASSET_ROOT/'MultiChat'
    elif args.multichat_annotations_root is not None:
        parser.error('--multichat-annotations-root requires the multichat profile.')
    if args.resume is not None and args.output_root is not None:
        parser.error('--resume uses the checkpoint run directory.')
    args.data_root = args.data_root or ASSET_ROOT/dataset
    values = {key: value.resolve() if isinstance(value, Path) else value for key, value in vars(args).items()}
    return TrainingConfig(**values, seed=seed, warmup=warmup, history_encode_batch_size=history_batch)
