"""YuE2 song generation for Apple Silicon Macs — Metal (MPS) and CPU."""
__version__ = "0.1.6"


def __getattr__(name):
    if name in {"is_apple_silicon", "require_apple_silicon"}:
        from . import _platform
        return getattr(_platform, name)
    if name in {"YuE2Pipeline", "SymbolicPlan", "SemanticResult", "SongResult"}:
        from . import pipeline
        return getattr(pipeline, name)
    if name in {"YuE2Config", "YuE2ForCausalLM"}:
        from . import modeling_yue2
        return getattr(modeling_yue2, name)
    if name in {"YuE2VAE", "YuE2VAEConfig"}:
        from . import modeling_vae
        return getattr(modeling_vae, name)
    if name == "parse_score":
        from .score import parse
        return parse
    if name == "StemSession":
        from .stems import StemSession
        return StemSession
    if name in {"derive_arrangement", "expand_leaves", "INSTRUMENT_CATALOG",
                "ENSEMBLE_PRESETS", "KIT_PRESETS"}:
        from . import arrange
        return getattr(arrange, name)
    if name == "render_session":
        from .mix import render_session
        return render_session
    if name == "deliver":
        from .deliver import deliver
        return deliver
    if name == "write_midi":
        from .midi import write_midi
        return write_midi
    raise AttributeError(name)
