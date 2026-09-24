"""Deterministic, token-weighted evaluation of a frozen checkpoint.

This is our block-context GPT-2 BPE protocol, NOT WikiText word-level leaderboard
perplexity. Every scored target is counted exactly once; no test-based selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model import ModelConfig, TransformerLM
from .data import verify


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def score_tokens(model, tokens, seq_len, batch_size=4, max_tokens=None, device='cpu'):
    """Return summed NLL and count; no repeated or omitted target in prefix."""
    if seq_len < 1 or batch_size < 1:
        raise ValueError('seq_len and batch_size must be positive')
    count = len(tokens) - 1
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        count = min(count, max_tokens)
    if count < 1:
        raise ValueError('Need at least two tokens')
    total = 0.0
    scored = 0
    model.eval()
    starts = list(range(0, count, seq_len))
    for offset in range(0, len(starts), batch_size):
        group = starts[offset:offset + batch_size]
        full = [s for s in group if s + seq_len <= count]
        tail = [s for s in group if s + seq_len > count]
        for members, length in [(full, seq_len)] + [([s], count-s) for s in tail]:
            if not members:
                continue
            block = np.stack([np.asarray(tokens[s:s+length+1], dtype=np.int64) for s in members])
            tensor = torch.as_tensor(block, device=device)
            logits = model(tensor[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   tensor[:, 1:].reshape(-1), reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite evaluation loss')
            total += float(loss.double())
            scored += len(members) * length
    assert scored == count
    return total, scored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-tokens', type=int)
    p.add_argument('--allow-test', action='store_true', help='Use only after the analysis protocol is frozen')
    a = p.parse_args()
    if a.split == 'test' and not a.allow_test:
        p.error('Freeze the analysis before opening test results; then pass --allow-test')
    if Path(a.out).exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**checkpoint['model_config'])
    model = TransformerLM(cfg, mode='standard').to(device)
    model.load_state_dict(checkpoint['model'])
    path = Path(a.data_dir) / f'{a.split}.bin'
    manifest_path = Path(a.data_dir) / 'manifest.json'
    manifest = verify(a.data_dir)
    if manifest.get('vocab_size') != cfg.vocab_size:
        raise ValueError('Dataset and model vocabulary differ')
    if manifest.get('synthetic') or manifest.get('synthetic_only'):
        raise ValueError('Synthetic fixture is not a paper evaluation dataset')
    tokens = np.memmap(path, dtype='<u2', mode='r')
    if int(tokens.max()) >= cfg.vocab_size:
        raise ValueError('Dataset tokenizer vocabulary exceeds model vocabulary')
    total, n = score_tokens(model, tokens, cfg.max_seq_len, a.batch_size, a.max_tokens, device)
    result = {'schema_version': 1, 'checkpoint_sha256': sha256(a.checkpoint),
              'checkpoint_step': checkpoint['step'], 'seed': checkpoint['seed'],
              'configured_training_steps': checkpoint['config']['train']['steps'],
              'phase': checkpoint['config'].get('study', {}).get('phase', checkpoint['config'].get('phase', 'unspecified')),
              'config_sha256': checkpoint['config_sha256'],
              'core_config_sha256': checkpoint['core_config_sha256'],
              'initialization_sha256': checkpoint['initialization_sha256'],
              'source_training_precision': checkpoint['precision'],
              'mode': checkpoint['mode'], 'split': a.split, 'n_scored_tokens': n,
              'dataset': manifest['dataset'], 'dataset_revision': manifest.get('revision'),
              'nll_nats_per_token': total/n, 'bpe_perplexity': math.exp(min(700, total/n)),
              'token_file_sha256': sha256(path), 'data_manifest_sha256': sha256(manifest_path),
              'train_data_manifest_sha256': checkpoint['data_manifest_sha256'],
              'context_length': cfg.max_seq_len, 'protocol': 'nonoverlapping_targets_block_reset_gpt2_bpe',
              'device': str(device), 'dtype': 'float32', 'torch_version': torch.__version__,
              'full_split': n == len(tokens)-1}
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
