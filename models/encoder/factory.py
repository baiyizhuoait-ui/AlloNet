"""Encoder factory: one dispatch point, default behaviour unchanged.

Why a factory instead of editing light_encoder.py: G3 needs a second encoder
topology selectable from the config, but every existing config (and every
frozen result in experiments/) must keep building EXACTLY as before.  The
dispatch therefore keys off `cfg["arch"]` and defaults to the original
LightEncoder instance, so a config without an `arch` key takes the identical
code path it took before this file existed.

    arch absent / "dws" / "light"   -> LightEncoder            (original)
    arch "ir" / "inverted_residual" -> InvertedResidualEncoder (G3)

Proof of equivalence for the default path lives in
scripts/phase6_g3_arch_probe.py (`--equivalence`), which builds the same cfg
both ways and compares state_dict keys, parameter counts and a seeded forward
pass -- not just the isinstance check.
"""
from models.encoder.light_encoder import LightEncoder

_DWS_ALIASES = {"", "dws", "light", "light_encoder", "espnet"}
_IR_ALIASES = {"ir", "inverted_residual", "mbv2"}


def build_encoder(cfg):
    cfg = cfg or {}
    arch = str(cfg.get("arch", "")).strip().lower()
    if arch in _DWS_ALIASES:
        return LightEncoder(cfg)
    if arch in _IR_ALIASES:
        from models.encoder.ir_encoder import InvertedResidualEncoder
        return InvertedResidualEncoder(cfg)
    raise ValueError(
        "unknown encoder arch %r; known: %s"
        % (cfg.get("arch"), sorted(_DWS_ALIASES | _IR_ALIASES)))
