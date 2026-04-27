"""
Unit tests for :mod:`spino.circuit.composition_io` loader utilities.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from spino.circuit.composition_io import DeviceCheckpoint, load_fno_device
from spino.mosfet.bsim_parser import BSIMParser


def test_load_fno_device_raises_on_missing_dataset(tmp_path: Path) -> None:
    """
    A missing HDF5 normalization file must surface as :class:`FileNotFoundError`.
    """
    ckpt = tmp_path / "exists.pt"
    ckpt.write_bytes(b"")
    spec = DeviceCheckpoint(
        strategy_name="sky130_nmos",
        checkpoint_path=ckpt,
        dataset_path=tmp_path / "missing.h5",
        width_um=1.0,
        length_um=0.18,
        label="X",
    )
    with pytest.raises(FileNotFoundError, match="Normalization dataset"):
        load_fno_device(spec)


def test_load_fno_device_raises_on_missing_checkpoint(tmp_path: Path) -> None:
    """
    A missing ``.pt`` file must surface as :class:`FileNotFoundError`.
    """
    spec = DeviceCheckpoint(
        strategy_name="sky130_nmos",
        checkpoint_path=tmp_path / "missing.pt",
        dataset_path=tmp_path / "missing.h5",
        width_um=1.0,
        length_um=0.18,
        label="X",
    )
    with pytest.raises(FileNotFoundError, match="FNO checkpoint"):
        load_fno_device(spec)


def test_read_curated_physics_empty_raises() -> None:
    """
    When BSIM inspection returns no rows, loading must not proceed silently.
    """
    h5 = Path("/app/datasets/sky130_nmos_61k_plus_shortch_supp8k.h5")
    ckpt = Path("/app/spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt")
    if not h5.exists() or not ckpt.exists():
        pytest.skip("Production artifacts not present")
    spec = DeviceCheckpoint(
        strategy_name="sky130_nmos",
        checkpoint_path=ckpt,
        dataset_path=h5,
        width_um=6.0,
        length_um=0.18,
        label="Y",
    )
    with patch.object(BSIMParser, "inspect_model", lambda self, name, w, l: {}):
        with pytest.raises(RuntimeError, match="BSIMParser returned no parameters"):
            load_fno_device(spec)


def test_load_fno_device_maps_to_cpu() -> None:
    """
    ``map_location`` should place buffers on the requested device when possible.
    """
    h5 = Path("/app/datasets/sky130_nmos_61k_plus_shortch_supp8k.h5")
    ckpt = Path("/app/spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt")
    if not h5.exists() or not ckpt.exists():
        pytest.skip("Production artifacts not present")
    spec = DeviceCheckpoint(
        strategy_name="sky130_nmos",
        checkpoint_path=ckpt,
        dataset_path=h5,
        width_um=6.0,
        length_um=0.18,
        label="Z",
    )
    dev = load_fno_device(spec, map_location="cpu")
    assert dev.v_mean.device == torch.device("cpu")
