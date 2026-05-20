"""
Adapter that drops a trained :class:`spino.mosfet.model.MosfetMLP` into the
same composition path the FNO production checkpoints use.

The :class:`~spino.circuit.devices.FnoMosfetDevice` wrapper is model-class
agnostic: it normalises terminal voltages, calls ``self.model(v_norm,
physics_norm)``, and decodes arcsinh-mA back to amperes. Any module with a
``forward(v_terminals, physical_params)`` signature returning ``(B, 1, T)``
arcsinh-mA outputs is a drop-in replacement, and ``MosfetMLP`` is by
construction such a module (same training-label convention as the FNO
variants, same scale).

The motivation is the inverter-chain off-diagonal hypothesis test
documented in ``docs/inverter_chain.md`` and ``docs/results.md`` §Digital
circuits: the per-timestep MLP has structurally zero
``dI[t]/dV[t']`` for ``t' != t``, so composing it inside the chain solver
isolates whether the FNO's spurious off-diagonal Jacobian entries are
what stop the whole-window Newton from converging on digital
trajectories.

Usage::

    from spino.circuit.composition_mlp_adapter import (
        load_mlp_device,
        load_inverter_chain_mlp_devices,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from spino.circuit.composition_io import (
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_DATASET,
    DeviceCheckpoint,
    _ensure_path,
    _read_curated_physics,
    _read_normalization_stats,
    _restore_state_dict,
)
from spino.circuit.devices import FnoMosfetDevice
from spino.mosfet.device_strategy import DeviceStrategy
from spino.mosfet.gen_data import ParameterSchema
from spino.mosfet.model import MosfetMLP

__all__ = [
    "MlpArchitecture",
    "DEFAULT_NFET_MLP_H64_CHECKPOINT",
    "DEFAULT_NFET_MLP_H128_CHECKPOINT",
    "load_mlp_device",
    "load_inverter_chain_mlp_devices",
]


_MODELS_ROOT = Path("/app/spino/models/mosfet")

DEFAULT_NFET_MLP_H64_CHECKPOINT: Path = _MODELS_ROOT / "nfet" / "mosfet_mlp_baseline_XpV7KFHL.pt"
DEFAULT_NFET_MLP_H128_CHECKPOINT: Path = _MODELS_ROOT / "nfet" / "mosfet_mlp_h128_OZyiUsFA.pt"


@dataclass(frozen=True, slots=True)
class MlpArchitecture:
    """Immutable hyperparameter record for :class:`MosfetMLP`.

    Defaults match the production NFET MLP h64 baseline (``hidden_dim = 64``,
    ``embedding_dim = 16``). Override ``hidden_dim`` to load h128 or any other
    capacity variant.
    """

    hidden_dim: int = 64
    embedding_dim: int = 16
    n_hidden_layers: int = 3
    embedding_hidden_dim: int = 128


def _instantiate_mlp(architecture: MlpArchitecture) -> nn.Module:
    """Builds an empty :class:`MosfetMLP` matching ``architecture``."""
    return MosfetMLP(
        input_param_dim=ParameterSchema.input_dim(),
        embedding_dim=architecture.embedding_dim,
        hidden_dim=architecture.hidden_dim,
        n_hidden_layers=architecture.n_hidden_layers,
        embedding_hidden_dim=architecture.embedding_hidden_dim,
    )


def load_mlp_device(
    spec: DeviceCheckpoint,
    *,
    architecture: MlpArchitecture | None = None,
    map_location: str | torch.device = "cpu",
) -> FnoMosfetDevice:
    """Materialise a single :class:`FnoMosfetDevice` backed by a trained MLP.

    Mirrors :func:`spino.circuit.composition_io.load_fno_device` but
    instantiates :class:`MosfetMLP` instead of :class:`MosfetVCFiLMFNO`. The
    returned wrapper exposes the same ``drain_current(v_terminals)``
    primitive the FNO devices use, so the chain and OTA solvers consume it
    without modification.

    :param spec: Checkpoint, dataset, and geometry record.
    :param architecture: Optional MLP hyperparameter override. Defaults to
        :class:`MlpArchitecture` (``hidden_dim = 64``); use
        ``MlpArchitecture(hidden_dim=128)`` to load the h128 baseline.
    :param map_location: Torch device the tensors should land on.
    :return: Configured :class:`FnoMosfetDevice` with an MLP backbone.
    :raises FileNotFoundError: When the checkpoint or dataset is missing.
    """
    arch = architecture or MlpArchitecture()
    _ensure_path(spec.checkpoint_path, "MLP checkpoint")
    _ensure_path(spec.dataset_path, "Normalization dataset")
    strategy = DeviceStrategy.create(spec.strategy_name)
    v_mean, v_std, p_mean, p_std = _read_normalization_stats(spec.dataset_path)
    physics_raw = _read_curated_physics(strategy, spec.width_um, spec.length_um)
    model = _restore_state_dict(_instantiate_mlp(arch), spec.checkpoint_path, map_location)
    wrapper = FnoMosfetDevice(
        model=model,
        v_mean=v_mean,
        v_std=v_std,
        p_mean=p_mean,
        p_std=p_std,
        physics_raw=physics_raw,
        label=spec.label,
    )
    return wrapper.to(map_location)


def load_inverter_chain_mlp_devices(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    *,
    n_stages: int,
    nfet_w_um: float,
    nfet_l_um: float,
    pfet_w_um: float,
    pfet_l_um: float,
    nfet_checkpoint: Path,
    pfet_checkpoint: Path,
    nfet_dataset: Path = DEFAULT_NFET_DATASET,
    pfet_dataset: Path = DEFAULT_PFET_DATASET,
    architecture: MlpArchitecture | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[tuple[FnoMosfetDevice, ...], tuple[FnoMosfetDevice, ...]]:
    """Return matched ``(NFET, PFET)`` MLP-backed device tuples for an N-stage chain.

    The chain solver is structurally typed (it consumes any callable matching
    the :class:`FnoMosfetDevice` interface), so swapping in MLP-backed
    wrappers exercises the same composition path with off-diagonal-free
    autograd Jacobians.

    :param n_stages: Number of series inverters (``>= 1``).
    :param nfet_w_um: NFET width shared by all stages.
    :param nfet_l_um: NFET length.
    :param pfet_w_um: PFET width.
    :param pfet_l_um: PFET length.
    :param nfet_checkpoint: NFET MLP ``.pt``.
    :param pfet_checkpoint: PFET MLP ``.pt``.
    :param nfet_dataset: NFET HDF5 normalisation file.
    :param pfet_dataset: PFET HDF5 normalisation file.
    :param architecture: Optional MLP hyperparameter override.
    :param map_location: Torch device placement.
    :return: Tuple ``(nfets, pfets)`` each of length ``n_stages``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    nfets: list[FnoMosfetDevice] = []
    pfets: list[FnoMosfetDevice] = []
    for idx in range(n_stages):
        nfet_spec = DeviceCheckpoint(
            strategy_name="sky130_nmos",
            checkpoint_path=nfet_checkpoint,
            dataset_path=nfet_dataset,
            width_um=nfet_w_um,
            length_um=nfet_l_um,
            label=f"NFET_XN{idx + 1}",
        )
        pfet_spec = DeviceCheckpoint(
            strategy_name="sky130_pmos",
            checkpoint_path=pfet_checkpoint,
            dataset_path=pfet_dataset,
            width_um=pfet_w_um,
            length_um=pfet_l_um,
            label=f"PFET_XP{idx + 1}",
        )
        nfets.append(load_mlp_device(nfet_spec, architecture=architecture, map_location=map_location))
        pfets.append(load_mlp_device(pfet_spec, architecture=architecture, map_location=map_location))
    return tuple(nfets), tuple(pfets)
