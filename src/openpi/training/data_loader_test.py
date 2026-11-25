import dataclasses
import itertools

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


class _IndexDataset:
    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        return np.int64(index)


def _index_batches(loader, n):
    return [np.asarray(b) for b in itertools.islice(loader, n)]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_torch_data_loader_skip_batches(num_workers: int):
    # 1003 samples / 64 = 15 batches per epoch (43 dropped), so the skips below cover mid-epoch, epoch edges and
    # whole skipped epochs. A skipped loader must continue the unskipped stream, then keep matching for 2+ epochs.
    dataset = _IndexDataset(1003)
    kwargs = {"local_batch_size": 64, "shuffle": True, "num_workers": num_workers, "seed": 42, "framework": "pytorch"}
    n_check = 40
    skips = [1, 7, 14, 15, 16, 30, 47]
    ref = _index_batches(_data_loader.TorchDataLoader(dataset, **kwargs), max(skips) + n_check)
    assert len({tuple(b) for b in ref[:15]}) == 15  # Sanity: batches differ.
    for skip in skips:
        got = _index_batches(_data_loader.TorchDataLoader(dataset, **kwargs, skip_batches=skip), n_check)
        for i, (a, b) in enumerate(zip(ref[skip : skip + n_check], got, strict=True)):
            np.testing.assert_array_equal(a, b, err_msg=f"skip={skip} batch={i}")


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
