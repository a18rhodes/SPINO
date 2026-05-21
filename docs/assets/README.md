# Tracked artefact index

Every quantitative claim in the SPINO doc tree (`README.md`, `docs/results.md`,
`docs/sizing.md`, `docs/composition.md`, `docs/ota_composition.md`,
`docs/attribution.md`, `docs/inverter_chain.md`) resolves to a JSON or PNG
under `docs/assets/`. This file is the index: for each headline number or
figure, the table below points at the file that backs it.

Local re-run trees under `runs/` are git-ignored; the tracked copies here are
the canonical references that survive a clean clone.

## CS amplifier composition (Phase 3b)

| Claim | Backing file |
|---|---|
| L=0.18 stress CUDA: r=0.99748, max\|ΔV\|=25.78 mV, DC \|ΔV\|/VDD=0.92 % | [`cs_amp_fno_exp2/summary.json`](cs_amp_fno_exp2/summary.json) |
| L=0.40 showcase CUDA: r=0.99981, max\|ΔV\|=2.39 mV, transient R²=0.9946 | [`cs_amp_fno_l040_exp2/summary.json`](cs_amp_fno_l040_exp2/summary.json) |
| L=0.18 / L=0.40 CPU reruns (runtime context) | [`cs_amp_fno/summary.json`](cs_amp_fno/summary.json), [`cs_amp_fno_l040/summary.json`](cs_amp_fno_l040/summary.json) |
| L=0.18 SPICE reference (peak gain 2.05, Idd@Vin\*=128.5 µA) | [`cs_amp/summary.json`](cs_amp/summary.json) |
| L=0.40 SPICE reference (peak gain 1.42, Idd@Vin\*=15.6 µA) | [`cs_amp_l040/summary.json`](cs_amp_l040/summary.json) |
| L=0.18 stress attribution: KCL 19.6 µA/73 nA, ratio 268x, NR α=1.0, VTC NFET ratio 4551.5x | [`cs_amp_fno_exp2/attribution/attribution_result.json`](cs_amp_fno_exp2/attribution/attribution_result.json) |

## 5T OTA composition (Phase 3b-OTA)

| Claim | Backing file |
|---|---|
| L=0.40: r=0.9997, max\|ΔV\|=68.7 mV, slew 47.92/48.41 V/µs, NR=5/11 | [`ota_5t_fno_l040/summary.json`](ota_5t_fno_l040/summary.json) |
| L=0.50: r=0.9997, max\|ΔV\|=68.9 mV, slew 42.47/40.47 V/µs, NR=6/12 | [`ota_5t_fno_l050/summary.json`](ota_5t_fno_l050/summary.json) |
| Triode fine-tune: r=0.9997, max\|ΔV\|=61.0 mV, slew 49.25/48.41 V/µs | [`ota_5t_fno_l040_exp07/summary.json`](ota_5t_fno_l040_exp07/summary.json) |
| L=0.40 / L=0.50 SPICE characterisation: 7x7 (W_diff, W_mirror) sweeps | [`ota_5t_l040/summary.json`](ota_5t_l040/summary.json), [`ota_5t_l050/summary.json`](ota_5t_l050/summary.json) |

## Gradient-based OTA sizing (multi-spec, v3, n_θ = 5)

| Claim | Backing file |
|---|---|
| Adam 50-step trajectory (loss, slew, power, θ per step) | [`sizing/v3_jtheta_fix/trajectory.json`](sizing/v3_jtheta_fix/trajectory.json) |
| Final θ = (3.638, 3.606, 1.592, 0.308, 1.537) µm/V | [`sizing/v3_jtheta_fix/theta_final.json`](sizing/v3_jtheta_fix/theta_final.json) |
| SPICE validation at θ_final: slew 38.83 V/µs, I_tail 77.1 µA, gain 34.11, swing 0.977 V | [`sizing/v3_jtheta_fix/spice_validation/summary.json`](sizing/v3_jtheta_fix/spice_validation/summary.json) |
| Loss + slew curve, θ trajectory, FNO vs SPICE overlay | [`sizing/v3_jtheta_fix/loss_and_slew.png`](sizing/v3_jtheta_fix/loss_and_slew.png), [`sizing/v3_jtheta_fix/theta_trajectory.png`](sizing/v3_jtheta_fix/theta_trajectory.png), [`sizing/v3_jtheta_fix/fno_vs_spice.png`](sizing/v3_jtheta_fix/fno_vs_spice.png) |
| FNO/IFT vs FD-SPICE comparison overlay (multi-spec) | [`sizing/v3_jtheta_fix/comparison_loss_slew.png`](sizing/v3_jtheta_fix/comparison_loss_slew.png), [`sizing/v3_jtheta_fix/comparison_theta.png`](sizing/v3_jtheta_fix/comparison_theta.png) |

## Gradient-based OTA sizing (per-device L, v4, n_θ = 7)

| Claim | Backing file |
|---|---|
| FNO/IFT 50-step trajectory at n_θ = 7 | [`sizing/v4_nt7/trajectory.json`](sizing/v4_nt7/trajectory.json) |
| FNO/IFT final θ = (3.640, 3.562, 1.590, 0.180, 0.180, 0.318, 1.535) µm/V | [`sizing/v4_nt7/theta_final.json`](sizing/v4_nt7/theta_final.json) |
| FNO/IFT SPICE validation: slew 43.67 V/µs, gain 15.55, power 144.8 µW, swing 0.774 V | [`sizing/v4_nt7/spice_validation/summary.json`](sizing/v4_nt7/spice_validation/summary.json) |
| FD-SPICE 50-step trajectory at n_θ = 7 (400 SPICE sims, 8 per step) | [`sizing/v4_nt7/fd_spice/trajectory.json`](sizing/v4_nt7/fd_spice/trajectory.json) |
| FD-SPICE final θ = (3.582, 3.537, 1.598, 0.180, 0.180, 0.180, 1.494) µm/V | [`sizing/v4_nt7/fd_spice/theta_final.json`](sizing/v4_nt7/fd_spice/theta_final.json) |
| FD-SPICE SPICE validation: slew 51.16 V/µs, gain 14.92, power 190.1 µW, swing 0.759 V | [`sizing/v4_nt7/fd_spice/spice_validation/summary.json`](sizing/v4_nt7/fd_spice/spice_validation/summary.json) |
| Per-step + trajectory plots (7-panel θ) | [`sizing/v4_nt7/loss_and_slew.png`](sizing/v4_nt7/loss_and_slew.png), [`sizing/v4_nt7/theta_trajectory.png`](sizing/v4_nt7/theta_trajectory.png), [`sizing/v4_nt7/fno_vs_spice.png`](sizing/v4_nt7/fno_vs_spice.png) |
| FNO/IFT vs FD-SPICE overlay at n_θ = 7 | [`sizing/v4_nt7/comparison_loss_slew.png`](sizing/v4_nt7/comparison_loss_slew.png), [`sizing/v4_nt7/comparison_theta.png`](sizing/v4_nt7/comparison_theta.png) |
| FD-SPICE forward-vs-central control (first 10 steps, max θ drift 1.7 %, slew Δ +2.2 %) | [`sizing/v4_nt7/fd_spice_central/trajectory.json`](sizing/v4_nt7/fd_spice_central/trajectory.json), [`sizing/v4_nt7/fd_spice_central/fd_forward_vs_central.png`](sizing/v4_nt7/fd_spice_central/fd_forward_vs_central.png) |

## Performance-surrogate baseline (Uhlmann route)

| Claim | Backing file |
|---|---|
| SPICE training set: 1000 LHS samples, 100 % converged | [`uhlmann_surrogate/training_set/samples.json`](uhlmann_surrogate/training_set/samples.json) |
| Surrogate test R²: slew 0.9978, power 0.9986, swing 0.9771 | [`uhlmann_surrogate/surrogate/test_metrics.json`](uhlmann_surrogate/surrogate/test_metrics.json) |
| Held-out gradient R² overall 0.9638 with W_tail = 0.14 (load-bearing degradation) | [`uhlmann_surrogate/surrogate/test_metrics.json`](uhlmann_surrogate/surrogate/test_metrics.json) |
| Surrogate training history | [`uhlmann_surrogate/surrogate/train_history.json`](uhlmann_surrogate/surrogate/train_history.json) |
| Uhlmann Adam 50-step trajectory (< 1 s wall-clock) | [`uhlmann_surrogate/adam/trajectory.json`](uhlmann_surrogate/adam/trajectory.json) |
| Uhlmann final θ (L_mirror at upper bound 0.50 µm, slew 47.5 V/µs SPICE, power 175.6 µW SPICE, gain 23.0 V/V SPICE) | [`uhlmann_surrogate/adam/spice_validation/summary.json`](uhlmann_surrogate/adam/spice_validation/summary.json) |
| Three-way Adam overlay (FNO/IFT vs FD-SPICE vs Uhlmann) | [`sizing/v4_nt7/three_way/three_way_loss_slew_power.png`](sizing/v4_nt7/three_way/three_way_loss_slew_power.png), [`sizing/v4_nt7/three_way/three_way_theta.png`](sizing/v4_nt7/three_way/three_way_theta.png) |
| Uhlmann seed-variance: 5 surrogate seeds × 3 θ_init, L_mirror 8/15 lower bound, 5/15 upper bound, 2/15 interior, gain spread 12.1–27.4 V/V | [`uhlmann_surrogate/seed_variance/aggregate.json`](uhlmann_surrogate/seed_variance/aggregate.json), [`uhlmann_surrogate/seed_variance/seed_variance.png`](uhlmann_surrogate/seed_variance/seed_variance.png) |

## FNO safe operating region (per-device ratio error)

| Claim | Backing file |
|---|---|
| NFET safe region @ (W=6 µm, L=0.18 µm): τ=0.1 76 % / τ=0.3 87 % / τ=1.0 94 % coverage | [`safe_region/cs_amp_l018/summary.json`](safe_region/cs_amp_l018/summary.json) |
| PFET safe region @ (W=4.5 µm, L=0.18 µm): τ=0.1 50 % / τ=0.3 80 % / τ=1.0 91 % coverage | [`safe_region/cs_amp_l018/summary.json`](safe_region/cs_amp_l018/summary.json) |
| NFET / PFET heat maps with τ contours | [`safe_region/cs_amp_l018/nfet_core_L018_safe_region.png`](safe_region/cs_amp_l018/nfet_core_L018_safe_region.png), [`safe_region/cs_amp_l018/pfet_core_L018_safe_region.png`](safe_region/cs_amp_l018/pfet_core_L018_safe_region.png) |
| Raw 91x91 grids (V_g, V_d, ids_spice, ids_fno, err_ratio) | [`safe_region/cs_amp_l018/nfet_core_L018_grid.npz`](safe_region/cs_amp_l018/nfet_core_L018_grid.npz), [`safe_region/cs_amp_l018/pfet_core_L018_grid.npz`](safe_region/cs_amp_l018/pfet_core_L018_grid.npz) |

## Off-corner transferability spot check

| Claim | Backing file |
|---|---|
| FNO vs SPICE tt @ 27 °C: r=0.99966, max\|ΔV\|=68.7 mV, slew 48.28 V/µs | [`off_corner/summary.json`](off_corner/summary.json) |
| FNO vs SPICE ff @ 125 °C: r=0.99912, max\|ΔV\|=171.8 mV, slew 46.80 V/µs | [`off_corner/summary.json`](off_corner/summary.json) |
| FNO slew at design point: 48.29 V/µs | [`off_corner/summary.json`](off_corner/summary.json) |
| V_out three-way overlay (FNO + tt + ff) | [`off_corner/v_out_overlay.png`](off_corner/v_out_overlay.png) |

## Inverter chain regime boundary (digital negative result)

| Claim | Backing file |
|---|---|
| N=1: DC ✓ (0 iters), transient ✗ (25 cap), max\|ΔV\|=1.800 V | [`inv_chain/matrix/n1/rep00/summary.json`](inv_chain/matrix/n1/rep00/summary.json) |
| N=2: DC ✓ (3 iters), transient ✗ (25 cap), max\|ΔV\|=1.777 V | [`inv_chain/matrix/n2/rep00/summary.json`](inv_chain/matrix/n2/rep00/summary.json) |
| N=4: DC ✓ (5 iters), transient ✗ (25 cap), max\|ΔV\|=1.778 V | [`inv_chain/matrix/n4/rep00/summary.json`](inv_chain/matrix/n4/rep00/summary.json) |
| Aggregate across N | [`inv_chain/matrix/aggregate_summary.json`](inv_chain/matrix/aggregate_summary.json) |
| Per-stage final-output overlay vs SPICE | [`inv_chain/matrix/n{1,2,4}/rep00/final_output_overlay.png`](inv_chain/matrix/) |
| MLP-h64 composition at N=1 (DC non-converge, max\|ΔV\|=1.80 V, fake fixed point at mid-supply init) | [`inv_chain/matrix_mlp_h64/n1/rep00/summary.json`](inv_chain/matrix_mlp_h64/n1/rep00/summary.json) |
| MLP-h128 composition at N=1, 2, 4 (DC non-converge, transient stuck at IC) | [`inv_chain/matrix_mlp_h128/n{1,2,4}/rep00/summary.json`](inv_chain/matrix_mlp_h128/) |
| Jacobian probe at chain DC init (FNO vs MLP-h64 vs MLP-h128, N=2, V_in=0 rail init) | [`inv_chain/jacobian_probe/`](inv_chain/jacobian_probe/) |

## Standalone device operators

| Claim | Backing file |
|---|---|
| sky130 NFET production figures, transfer / output / subthreshold parity | [`mosfet/nfet/`](mosfet/nfet/) |
| sky130 PFET production figures | [`mosfet/pfet/`](mosfet/pfet/) |
| MLP ablation (h64 / h128 / FNO) on NFET dataset | [`mosfet/nfet/mlp_ablation/`](mosfet/nfet/mlp_ablation/) |
| Diode operator, single rectifier and adversarial parity | [`diode/`](diode/) |
| RC operator parity, chirp / noise / log-uniform | [`simple_rc/`](simple_rc/) |

## Notes on local re-run trees

Each entry above has a `runs/<topic>/...` analogue in the local tree that the
documented CLIs write to (`spino.circuit.compose`, `compose_ota`,
`compose_chain`, `characterize`, `characterize_ota`, `sizing`,
`off_corner_probe`, `plot_sizing_trajectory`, `plot_sizing_comparison`,
`ota_attribution`). The local trees are not tracked: they regenerate from
the reproduction commands listed alongside each results section.
