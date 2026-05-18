import logging
import os
from typing import Dict, Tuple, Iterator, Union, Any, List

import torch as tr
import torch.nn.functional as F
from scipy.stats import loguniform
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ReadOnlyTensorDict(nn.Module):
    def __init__(self, data: Dict[str | int, T], persistent: bool = True):
        super().__init__()
        self.persistent = persistent
        self.keys = set(data.keys())
        for k, v in data.items():
            self.register_buffer(f"tensor_{k}", v, persistent=persistent)

    def __getitem__(self, key: str | int) -> T:
        return self.get_buffer(f"tensor_{key}")

    def __contains__(self, key: str | int) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.keys)

    def keys(self) -> Iterator[str | int]:
        return iter(self.keys)

    def values(self) -> Iterator[T]:
        for k in self.keys:
            yield self[k]

    def items(self) -> Iterator[Tuple[str | int, T]]:
        for k in self.keys:
            yield k, self[k]


def linear_interpolate_last_dim(x: T, n: int, align_corners: bool = True) -> T:
    n_dim = x.ndim
    assert 1 <= n_dim <= 3
    if x.size(-1) == n:
        return x
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        x = x.unsqueeze(1)
    x = F.interpolate(x, n, mode="linear", align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    return x


def choice(items: List[Any]) -> Any:
    assert len(items) > 0
    idx = randint(0, len(items))
    return items[idx]


def randint(low: int, high: int, n: int = 1) -> Union[int, T]:
    x = tr.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x


def sample_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    x = (tr.rand(n) * (high - low)) + low
    if n == 1:
        return x.item()
    return x


def sample_log_uniform(low: float, high: float, n: int = 1) -> Union[float, T]:
    # TODO(cm): replace with torch
    if low == high:
        if n == 1:
            return low
        else:
            return tr.full(size=(n,), fill_value=low)
    x = loguniform.rvs(low, high, size=n)
    if n == 1:
        return float(x)
    return tr.from_numpy(x)
