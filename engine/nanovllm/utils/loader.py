import os
from collections.abc import Iterable
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_weights(model: nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]):
    """Copy HF-named tensors into the (packed, TP-sharded) engine model in place.

    Used both for loading checkpoints and for syncing a trainer's live weights
    into the rollout engine (in-place copies keep CUDA-graph pointers valid).
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for weight_name, tensor in named_tensors:
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)


def load_model(model: nn.Module, path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            load_weights(model, ((name, f.get_tensor(name)) for name in f.keys()))
