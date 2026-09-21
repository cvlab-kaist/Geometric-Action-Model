"""Export trusted training checkpoints; load tensor-only public bundles."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from .contract import FORMAT, release_config, validate_config, embodiment_for_key

COMPONENTS = ("student_da3", "future_predictor", "action_head")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_weights(state):
    import torch
    result = {}
    for key, value in state.items():
        name = key.replace('_orig_mod.', '')
        if name in result or not isinstance(value, torch.Tensor):
            raise ValueError('Duplicate compiled key or non-tensor model state')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint parameter: {name}')
        result[name] = value.detach().cpu()
    return result


def export_checkpoint(checkpoint, output, *, text_revision):
    """Maintainer-only: input uses pickle and MUST be an owned, trusted checkpoint."""
    import torch
    from .normalization import Normalizer
    if not re.fullmatch(r'[0-9a-f]{40}', text_revision):
        raise ValueError('Pin text_revision to the 40-character T5 Hub commit used for training')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new bundle directory; exports never overwrite artifacts')
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = release_config(raw['config'], raw['train_steps'])
    config['text_revision'] = text_revision
    validate_config(config)
    if raw.get('proprio_head') is not None or raw.get('proprio_conditioner') is not None:
        raise ValueError('Unexpected extra state conditioning head')
    weights = {name: clean_weights(raw[name]) for name in COMPONENTS}
    if any('single_arm_proprio_proj.' in k for k in weights['future_predictor']):
        raise ValueError('A separate state-projection checkpoint is not this shared 16D release')
    statistics = {}
    for public_name, key in (('action', 'action_normalizer'), ('state', 'proprio_normalizer')):
        source = raw[key]
        statistics[public_name] = {
            'norm_mode': source['norm_mode'], 'eps': float(source['eps']),
            'stats_by_key': {str(k): {field: torch.as_tensor(row[field]).tolist()
                                    for field in ('q01', 'q99', 'mask')}
                             for k, row in source['stats_by_key'].items()},
        }
        Normalizer(statistics[public_name])
        for k in statistics[public_name]['stats_by_key']:
            embodiment_for_key(k)
    if statistics['action']['stats_by_key'].keys() != statistics['state']['stats_by_key'].keys():
        raise ValueError('Action/state normalization keys differ')
    output.mkdir(parents=True)
    torch.save(weights, output/'model.pt')
    (output/'normalization.json').write_text(json.dumps(statistics, indent=2, allow_nan=False)+'\n')
    (output/'config.json').write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
    # Written last. No source path, optimizer, dataset provenance, tokens or run IDs.
    manifest = {'format': FORMAT, 'files': {name: sha256(output/name)
                for name in ('model.pt', 'config.json', 'normalization.json')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_bundle(directory):
    import torch
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('format') != FORMAT or set(manifest.get('files', {})) != {'model.pt','config.json','normalization.json'}:
        raise ValueError('Incomplete or unsupported mobile bundle')
    for name, expected in manifest['files'].items():
        if sha256(directory/name) != expected:
            raise ValueError(f'Bundle checksum mismatch: {name}')
    config = json.loads((directory/'config.json').read_text());validate_config(config)
    statistics = json.loads((directory/'normalization.json').read_text())
    from .normalization import Normalizer
    if set(statistics) != {'action', 'state'}:
        raise ValueError('Missing or unexpected normalization components')
    for state in statistics.values():
        Normalizer(state)
        for key in state['stats_by_key']:
            embodiment_for_key(key)
    if statistics['action']['stats_by_key'].keys() != statistics['state']['stats_by_key'].keys():
        raise ValueError('Action/state normalization keys differ')
    weights = torch.load(directory/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    if set(weights) != set(COMPONENTS):
        raise ValueError('Missing or unexpected model components')
    return config, statistics, weights
