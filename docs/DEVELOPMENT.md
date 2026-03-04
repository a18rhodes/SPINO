# Development Guide

Instructions for reproducing results, extending the codebase, and contributing to SPINO.

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| Python 3.11 | Runtime |
| PyTorch >= 2.9 | Backend |
| NGSPICE | SPICE simulation backend |
| Sky130 PDK (via Volare) | MOSFET model cards |
| Poetry | Dependency management |

The included Dockerfile and dev container configuration handle all of the above automatically.

---

## Installation

```bash
# Inside the dev container (recommended):
pip install -e .

# Or via Poetry directly:
poetry install
```

The `scripts/setup_pdk.sh` script installs the SkyWater 130 nm PDK via Volare.
This runs automatically during container build.

---

## Project Layout

```
spino/
    mosfet/         NMOS operator (VCFiLM-FNO)
    rc/             RC circuit operator (dimensionless FNO)
    diode/          Diode operator (direct-injection FNO)
    models/         Production checkpoints (per-device subdirectories)
    export_docs_figures.py   CLI for generating documentation figures
datasets/           HDF5 training and evaluation datasets
docs/               Technical documentation and figure assets
```

---

## Generating Documentation Figures

All publication figures (light background) are generated from a single entry point:

```bash
python -m spino.export_docs_figures
```

This writes PNGs into `docs/assets/{mosfet,simple_rc,diode}/` and requires the production
checkpoints to be present under `spino/models/`.

**Selective generation:**

```bash
python -m spino.export_docs_figures --no-mosfet    # Skip MOSFET (slow SPICE validation)
python -m spino.export_docs_figures --no-rc         # Skip RC
python -m spino.export_docs_figures --no-diode      # Skip diode
```

**Custom output directory:**

```bash
python -m spino.export_docs_figures --docs-assets path/to/output
```

---

## Running Evaluation

### MOSFET

```bash
python -m spino.mosfet.run_evaluation \
    --model-path spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt \
    --dataset-path datasets/sky130_nmos_61k_plus_shortch_supp8k.h5
```

This runs the full evaluation suite: core IV sweeps against live NGSPICE, comprehensive
multi-geometry assessment, and summary metrics. Output figures and tables go to `spino/runs/mosfet/`.

### RC Circuit

```bash
python -m spino.rc.evaluate
```

### Diode

```bash
python -m spino.diode.evaluate
```

---

## Training

### MOSFET

```bash
python -m spino.mosfet.train
```

Training configuration is managed through `spino/config.py` and command-line arguments.
The production model was trained in two phases (see [docs/nfet.md](nfet.md) for methodology).

### RC Circuit

```bash
python -m spino.rc.train
```

### Diode

```bash
python -m spino.diode.train
```

---

## Production Checkpoints

| Device | Checkpoint | Notes |
|---|---|---|
| MOSFET | `mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt` | VCFiLM-FNO, Phase 2 fine-tune |
| RC | `Dimensionless_With_Gaussian_Noise_And_Chirp_Log_Uniform_eyJhZGFt.pt` | Spectral-augmented |
| Diode | `diode_spice_supervised_ntvnY8cj.pt` | NGSPICE-supervised |

Legacy and experimental checkpoints are retained in `spino/models/` for reproducibility
but are not used by the documentation figure pipeline.

---

## Datasets

All training data is stored as HDF5 under `datasets/`. The production MOSFET dataset is
`sky130_nmos_61k_plus_shortch_supp8k.h5` (61,041 samples). Additional datasets from
ablation studies (geometry-specific, subthreshold-focused, cross-bin) are retained for
reproducibility.

RC and diode datasets are generated on-the-fly from their respective NGSPICE/solver backends.

---

## Quality Assurance

```bash
# Formatting
black --line-length 120 .

# Linting
pylint spino/

# Tests with coverage
pytest --cov=. --cov-fail-under=100
```

---

## Experiment Logs

Detailed experiment histories, ablation results, and architectural decisions are documented in:

- [spino/mosfet/CURRENT_STATUS.md](../spino/mosfet/CURRENT_STATUS.md) — chronological experiment log
- [spino/mosfet/NEXT_STEPS.md](../spino/mosfet/NEXT_STEPS.md) — design rationale and future plans
