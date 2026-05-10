"""
Loaders for the neural-composition CS amplifier solver.

Given two production FNO checkpoints (one NFET, one PFET) and the datasets
they were trained against, ``load_cs_amp_devices`` materializes a pair of
:class:`spino.circuit.devices.FnoMosfetDevice` instances ready to be plugged
into the Newton-Raphson solver in :mod:`spino.circuit.composition`.

The loader is the only place in the composition pipeline that touches disk:
it parses the BSIM model card via :class:`spino.mosfet.bsim_parser.BSIMParser`,
opens the HDF5 dataset to recover the dataset-specific z-score statistics,
restores the FNO state dictionary, and fails loudly when any ingredient is
missing. After this returns, the caller holds a fully self-contained pair
of differentiable wrappers that can be moved to a CUDA device with
``.to(device)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from spino.circuit.devices import FnoMosfetDevice
from spino.mosfet.bsim_parser import BSIMParser
from spino.mosfet.device_strategy import DeviceStrategy
from spino.mosfet.gen_data import ParameterSchema, PreGeneratedMosfetDataset
from spino.mosfet.model import MosfetVCFiLMFNO

__all__ = [
    "DeviceCheckpoint",
    "ModelArchitecture",
    "load_cs_amp_devices",
    "load_fno_device",
    "load_inverter_chain_devices",
    "load_ota_5t_devices",
]

DEFAULT_NFET_CHECKPOINT = Path("/app/spino/models/mosfet/nfet/mosfet_vcfilm_exp20_float_src_68SXXICB.pt")
DEFAULT_PFET_CHECKPOINT = Path("/app/spino/models/mosfet/pfet/mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt")
DEFAULT_NFET_DATASET = Path("/app/datasets/sky130_nmos_v2_76k_float_src.h5")
DEFAULT_PFET_DATASET = Path("/app/datasets/sky130_pmos_48k_sweep_aug.h5")


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """
    Immutable record of the FNO architecture hyperparameters.

    Defaults match the production VCFiLM checkpoints (Exp 19b, Exp 06).

    :param modes: Number of Fourier modes retained per spectral conv.
    :param width: Hidden channel width of the FNO blocks.
    :param embedding_dim: Latent dimension of the physics embedding MLP.
    """

    modes: int = 256
    width: int = 64
    embedding_dim: int = 16


@dataclass(frozen=True, slots=True)
class DeviceCheckpoint:
    """
    Inputs needed to materialize one :class:`FnoMosfetDevice`.

    :param strategy_name: Registered :class:`DeviceStrategy` key
        (``"sky130_nmos"`` or ``"sky130_pmos"``).
    :param checkpoint_path: Filesystem path to the trained ``.pt`` file.
    :param dataset_path: HDF5 dataset path used to recover normalization
        statistics. Must match the dataset the checkpoint was trained
        against, otherwise the model will silently produce wrong currents.
    :param width_um: Channel width in microns.
    :param length_um: Channel length in microns.
    :param label: Human-readable identifier for logs and reprs.
    """

    strategy_name: str
    checkpoint_path: Path
    dataset_path: Path
    width_um: float
    length_um: float
    label: str


def _ensure_path(path: Path, kind: str) -> Path:
    """
    Raises ``FileNotFoundError`` if a required path is missing.

    :param path: Filesystem path to validate.
    :param kind: Short descriptor used in the error message.
    :return: The validated path.
    :raises FileNotFoundError: When the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def _read_normalization_stats(
    dataset_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Opens the HDF5 dataset and extracts dataset-specific z-score statistics.

    :param dataset_path: Path to a pre-generated MOSFET HDF5 file.
    :return: Tuple ``(v_mean, v_std, p_mean, p_std)`` where ``v_*`` are
        ``(4, 1)`` tensors and ``p_*`` are 1-D tensors of length 29.
    """
    with PreGeneratedMosfetDataset(str(dataset_path), normalize=True, use_curated_params=True) as dataset:
        return (
            dataset.voltages_mean.clone(),
            dataset.voltages_std.clone(),
            dataset.physics_mean.clone(),
            dataset.physics_std.clone(),
        )


def _read_curated_physics(strategy: DeviceStrategy, width_um: float, length_um: float) -> torch.Tensor:
    """
    Extracts the curated 29-parameter BSIM tensor for the device geometry.

    :param strategy: Device strategy carrying ``model_name`` and ``pdk_root``.
    :param width_um: Channel width in microns.
    :param length_um: Channel length in microns.
    :return: Tensor of shape ``(29,)`` in
        :data:`ParameterSchema.TRAINING_KEYS` order.
    """
    parser = BSIMParser(pdk_root=strategy.pdk_root)
    raw_params = parser.inspect_model(strategy.model_name, w=str(width_um), l=str(length_um))
    if not raw_params:
        raise RuntimeError(f"BSIMParser returned no parameters for {strategy.model_name} (W={width_um}, L={length_um})")
    full_tensor = ParameterSchema.to_tensor(raw_params).squeeze()
    params_dict = {key: full_tensor[i].item() for i, key in enumerate(ParameterSchema.SUPPORTED_KEYS)}
    return ParameterSchema.to_training_tensor(params_dict)


def _instantiate_model(architecture: ModelArchitecture) -> nn.Module:
    """
    Builds an empty ``MosfetVCFiLMFNO`` matching the production checkpoints.

    :param architecture: Architecture hyperparameters.
    :return: Untrained model ready for ``load_state_dict``.
    """
    return MosfetVCFiLMFNO(
        input_param_dim=ParameterSchema.input_dim(),
        embedding_dim=architecture.embedding_dim,
        modes=architecture.modes,
        width=architecture.width,
    )


def _restore_state_dict(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device) -> nn.Module:
    """
    Loads a checkpoint into ``model`` and switches the module to eval mode.

    :param model: Untrained model with the matching architecture.
    :param checkpoint_path: ``.pt`` file produced by the training pipeline.
    :param map_location: Torch device to materialize tensors on.
    :return: The same model instance with weights restored.
    """
    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_fno_device(
    spec: DeviceCheckpoint,
    *,
    architecture: ModelArchitecture | None = None,
    map_location: str | torch.device = "cpu",
) -> FnoMosfetDevice:
    """
    Materializes a single :class:`FnoMosfetDevice` from a checkpoint spec.

    :param spec: Bundle describing the checkpoint, dataset, and geometry.
    :param architecture: Optional override for the FNO architecture
        hyperparameters; defaults to the production VCFiLM configuration.
    :param map_location: Torch device the tensors should land on.
    :return: Configured :class:`FnoMosfetDevice`.
    :raises FileNotFoundError: When the checkpoint or dataset is missing.
    """
    arch = architecture or ModelArchitecture()
    _ensure_path(spec.checkpoint_path, "FNO checkpoint")
    _ensure_path(spec.dataset_path, "Normalization dataset")
    strategy = DeviceStrategy.create(spec.strategy_name)
    v_mean, v_std, p_mean, p_std = _read_normalization_stats(spec.dataset_path)
    physics_raw = _read_curated_physics(strategy, spec.width_um, spec.length_um)
    model = _restore_state_dict(_instantiate_model(arch), spec.checkpoint_path, map_location)
    device_wrapper = FnoMosfetDevice(
        model=model,
        v_mean=v_mean,
        v_std=v_std,
        p_mean=p_mean,
        p_std=p_std,
        physics_raw=physics_raw,
        label=spec.label,
    )
    return device_wrapper.to(map_location)


def load_cs_amp_devices(  # pylint: disable=too-many-arguments
    *,
    nfet_w_um: float,
    nfet_l_um: float,
    pfet_w_um: float,
    pfet_l_um: float,
    nfet_checkpoint: Path = DEFAULT_NFET_CHECKPOINT,
    pfet_checkpoint: Path = DEFAULT_PFET_CHECKPOINT,
    nfet_dataset: Path = DEFAULT_NFET_DATASET,
    pfet_dataset: Path = DEFAULT_PFET_DATASET,
    architecture: ModelArchitecture | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[FnoMosfetDevice, FnoMosfetDevice]:
    """
    Loads the NFET and PFET FNOs needed for the CS amplifier composition.

    :param nfet_w_um: NFET channel width in microns.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_w_um: PFET channel width in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param nfet_checkpoint: NFET ``.pt`` file (Exp 19b production by default).
    :param pfet_checkpoint: PFET ``.pt`` file (Exp 06 production by default).
    :param nfet_dataset: NFET HDF5 dataset for normalization statistics.
    :param pfet_dataset: PFET HDF5 dataset for normalization statistics.
    :param architecture: Optional architecture override (must match training).
    :param map_location: Torch device for the loaded tensors.
    :return: Tuple ``(nfet_device, pfet_device)`` of
        :class:`FnoMosfetDevice` ready for composition.
    """
    nfet_spec = DeviceCheckpoint(
        strategy_name="sky130_nmos",
        checkpoint_path=Path(nfet_checkpoint),
        dataset_path=Path(nfet_dataset),
        width_um=nfet_w_um,
        length_um=nfet_l_um,
        label="NFET",
    )
    pfet_spec = DeviceCheckpoint(
        strategy_name="sky130_pmos",
        checkpoint_path=Path(pfet_checkpoint),
        dataset_path=Path(pfet_dataset),
        width_um=pfet_w_um,
        length_um=pfet_l_um,
        label="PFET",
    )
    nfet_device = load_fno_device(nfet_spec, architecture=architecture, map_location=map_location)
    pfet_device = load_fno_device(pfet_spec, architecture=architecture, map_location=map_location)
    return nfet_device, pfet_device


def load_inverter_chain_devices(
    *,
    n_stages: int,
    nfet_w_um: float,
    nfet_l_um: float,
    pfet_w_um: float,
    pfet_l_um: float,
    nfet_checkpoint: Path = DEFAULT_NFET_CHECKPOINT,
    pfet_checkpoint: Path = DEFAULT_PFET_CHECKPOINT,
    nfet_dataset: Path = DEFAULT_NFET_DATASET,
    pfet_dataset: Path = DEFAULT_PFET_DATASET,
    architecture: ModelArchitecture | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[tuple[FnoMosfetDevice, ...], tuple[FnoMosfetDevice, ...]]:
    """
    Loads ``N`` NFET and ``N`` PFET FNO wrappers for a matched inverter chain.

    Every stage reuses the same ``(W,L)`` sizing; each device is an independent
    :class:`~spino.circuit.devices.FnoMosfetDevice` instance (no weight sharing).

    :param n_stages: Number of series inverters (>= 1).
    :param nfet_w_um: NFET width shared by all stages.
    :param nfet_l_um: NFET length.
    :param pfet_w_um: PFET width.
    :param pfet_l_um: PFET length.
    :param nfet_checkpoint: NFET ``.pt``.
    :param pfet_checkpoint: PFET ``.pt``.
    :param nfet_dataset: NFET HDF5 normalisation file.
    :param pfet_dataset: PFET HDF5 normalisation file.
    :param architecture: Optional FNO hyperparameter override.
    :param map_location: Torch device placement.
    :return: Tuple ``(nfets, pfets)`` each of length ``n_stages``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    nfets_list: list[FnoMosfetDevice] = []
    pfets_list: list[FnoMosfetDevice] = []
    for idx in range(n_stages):
        n_dev, p_dev = load_cs_amp_devices(
            nfet_w_um=nfet_w_um,
            nfet_l_um=nfet_l_um,
            pfet_w_um=pfet_w_um,
            pfet_l_um=pfet_l_um,
            nfet_checkpoint=nfet_checkpoint,
            pfet_checkpoint=pfet_checkpoint,
            nfet_dataset=nfet_dataset,
            pfet_dataset=pfet_dataset,
            architecture=architecture,
            map_location=map_location,
        )
        n_dev.label = f"NFET_XN{idx + 1}"
        p_dev.label = f"PFET_XP{idx + 1}"
        nfets_list.append(n_dev)
        pfets_list.append(p_dev)
    return tuple(nfets_list), tuple(pfets_list)


def load_ota_5t_devices(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    *,
    diff_w_um: float,
    diff_l_um: float,
    mirror_w_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    nfet_checkpoint: Path = DEFAULT_NFET_CHECKPOINT,
    pfet_checkpoint: Path = DEFAULT_PFET_CHECKPOINT,
    nfet_dataset: Path = DEFAULT_NFET_DATASET,
    pfet_dataset: Path = DEFAULT_PFET_DATASET,
    architecture: ModelArchitecture | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[FnoMosfetDevice, FnoMosfetDevice, FnoMosfetDevice, FnoMosfetDevice, FnoMosfetDevice]:
    """
    Loads the five FNO device wrappers for a 5T OTA.

    Returns independent :class:`FnoMosfetDevice` instances (no weight sharing)
    for M1, M2 (NFET diff pair), M3, M4 (PFET current mirror), and M5
    (NFET tail current source).

    :param diff_w_um: Diff-pair width (M1, M2) in microns.
    :param diff_l_um: Diff-pair channel length in microns.
    :param mirror_w_um: Current-mirror width (M3, M4) in microns.
    :param mirror_l_um: Current-mirror channel length in microns.
    :param tail_w_um: Tail current-source width (M5) in microns.
    :param tail_l_um: Tail current-source channel length in microns.
    :param nfet_checkpoint: NFET ``.pt`` checkpoint (Exp 20 float-source by default).
    :param pfet_checkpoint: PFET ``.pt`` checkpoint (Exp 06 by default).
    :param nfet_dataset: NFET HDF5 dataset for normalization statistics.
    :param pfet_dataset: PFET HDF5 dataset for normalization statistics.
    :param architecture: Optional FNO architecture override.
    :param map_location: Torch device placement.
    :return: Tuple ``(M1, M2, M3, M4, M5)`` of :class:`FnoMosfetDevice`.
    """

    def _nfet(label: str, w: float, l: float) -> FnoMosfetDevice:
        spec = DeviceCheckpoint("sky130_nmos", nfet_checkpoint, nfet_dataset, w, l, label)
        return load_fno_device(spec, architecture=architecture, map_location=map_location)

    def _pfet(label: str, w: float, l: float) -> FnoMosfetDevice:
        spec = DeviceCheckpoint("sky130_pmos", pfet_checkpoint, pfet_dataset, w, l, label)
        return load_fno_device(spec, architecture=architecture, map_location=map_location)

    return (
        _nfet("M1_nfet", diff_w_um, diff_l_um),
        _nfet("M2_nfet", diff_w_um, diff_l_um),
        _pfet("M3_pfet", mirror_w_um, mirror_l_um),
        _pfet("M4_pfet", mirror_w_um, mirror_l_um),
        _nfet("M5_nfet", tail_w_um, tail_l_um),
    )
