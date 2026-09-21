"""Encoder factory: one dispatch point, default behaviour unchanged.

G3 needs a second encoder topology selectable from the config, but every
frozen result must keep building exactly as before. Dispatch keys off
cfg["arch"] and defaults to LightEncoder, so a config without an `arch` key
takes its original code path (equivalence proven by
scripts/phase6_g3_arch_probe.py --equivalence: state_dict keys, param count,
seeded forward).

    arch absent / "dws" / "light"   -> LightEncoder            (original)
    arch "ir" / "inverted_residual" -> InvertedResidualEncoder (G3)
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
