"""
Shared colour palettes for all Spino evaluation plots.

Two palettes are defined: one for dark-background figures (TensorBoard / training
monitoring) and one for light-background figures (documentation / publication).
All evaluation modules should import from here rather than defining colours inline.
"""

__all__ = ["DARK_PALETTE", "LIGHT_PALETTE", "get_palette", "coerce_palette"]

DARK_PALETTE: dict[str, str] = {
    "gt": "#ffffff",
    "pred": "#00ffff",
    "pred_sweep": "#ff00ff",
    "pred_random": "#00ff00",
    "err_abs": "#ff6b6b",
    "err_rel": "#4ecdc4",
    "vg": "#00ff00",
    "vd": "#ff00ff",
    "vs": "#ffff00",
    "vth_line": "#ffaa00",
    "current": "#ff6600",
    "title": "white",
    "parity_scatter": "#00ffff",
    "parity_edge": "white",
    "snapshot_true": "#ffffff",
    "snapshot_pred": "#00ffff",
    "ramp_parity": "#00ffff",
    "sweep_parity": "#ff00ff",
    "random_parity": "#00ff00",
    "voltage_overlay_vg": "#ffff00",
    "voltage_overlay_vd": "#ff6600",
    "voltage_overlay_label": "#ffff00",
    "suptitle": "white",
}

LIGHT_PALETTE: dict[str, str] = {
    "gt": "#1a1a1a",
    "pred": "#0066cc",
    "pred_sweep": "#9900aa",
    "pred_random": "#006600",
    "err_abs": "#cc0000",
    "err_rel": "#006b68",
    "vg": "#006600",
    "vd": "#9900aa",
    "vs": "#887700",
    "vth_line": "#cc7700",
    "current": "#884400",
    "title": "black",
    "parity_scatter": "#0066cc",
    "parity_edge": "#333333",
    "snapshot_true": "#1a1a1a",
    "snapshot_pred": "#0066cc",
    "ramp_parity": "#0066cc",
    "sweep_parity": "#9900aa",
    "random_parity": "#006600",
    "voltage_overlay_vg": "#006600",
    "voltage_overlay_vd": "#884400",
    "voltage_overlay_label": "#006600",
    "suptitle": "black",
}


def get_palette(dark: bool) -> dict[str, str]:
    """
    Returns the appropriate colour palette for plot rendering.

    :param dark: When True returns the dark-background palette suitable for
        TensorBoard and training monitors; when False returns the light-background
        palette suitable for documentation and publication.
    :return: A mapping of semantic colour keys to hex colour strings.
    """
    return DARK_PALETTE if dark else LIGHT_PALETTE


def coerce_palette(palette: dict[str, str] | None, dark: bool = True) -> dict[str, str]:
    """
    Resolves an optional palette argument to a concrete palette dict.

    Intended for private plot helpers that accept ``palette=None`` as a sentinel.
    Callers that already know the value of ``dark`` should use
    :func:`get_palette` directly.

    :param palette: An explicitly provided palette dict, or ``None``.
    :param dark: When *palette* is ``None``, selects the dark (``True``) or
        light (``False``) palette.
    :return: A concrete colour palette dict.
    """
    return palette if palette is not None else get_palette(dark)
