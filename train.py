import os

from catmer.config import parse_args


if __name__ == '__main__':
    config = parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from catmer.training import train

    train(config)
