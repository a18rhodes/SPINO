from pathlib import Path

MODELS_ROOT = Path("/app/spino/models")
FIGURES_ROOT = Path("/app/spino/figures")
RUNS_ROOT = Path("/app/spino/runs")

# Scale factor for arcsinh current transform (in mA units).
# arcsinh(I_mA / ARCSINH_SCALE_MA) compresses dynamic range while preserving sign.
# At 1e-6 mA (= 1 nA), subthreshold currents enter the logarithmic regime of
# arcsinh, compressing the 10^6 dynamic range to ~2:1 in transformed space
# (subth ~0.75, sat ~16). This was validated in Experiment 10 and must not be
# changed. Lowering to 1e-8 (Experiment 14) caused saturation sweep collapse
# by over-weighting the pA noise floor at the expense of above-threshold physics.
ARCSINH_SCALE_MA = 1e-6
