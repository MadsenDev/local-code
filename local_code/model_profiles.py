"""Model capability tiers and the recommended-minimum standard.

The reference hardware target is a single **NVIDIA RTX 3060 12 GB** running
models at roughly Q4_K_M. Footprint and reliability are estimated from the
model's *total* parameter count (parsed from the Ollama tag) plus a coarse
family-quality signal: instruct/coder-tuned families drive the tool loop far
more reliably than base or tiny chat models.

Tiers
-----
- ``recommended``  : the sweet spot. Fits with full context, fast, drives the
                     JSON tool loop reliably. This is the "works properly" bar.
- ``supported``    : works well, but either near the VRAM ceiling (slower,
                     less context headroom) or a touch less reliable on
                     multi-file edits.
- ``best_effort``  : runs, good for chat and inspection, flaky on multi-step
                     edits. Below the recommended standard.
- ``unsupported``  : does not fit the 3060 12 GB at a usable quant (will spill
                     to CPU and crawl), or too small to drive tools at all.

Nothing here blocks a model from running; it only sets expectations and tunes
generation defaults (few-shot examples, prompt trimming) per tier.
"""

import re
from dataclasses import dataclass, field

from .config import DEFAULT_NUM_CTX

# Families we trust to drive the tool loop, in rough descending order of how
# reliably they emit structured tool calls at a given size. Coder-tuned
# families get a reliability bump because the backend loop is code-shaped.
CODER_FAMILIES = (
    "qwen2.5-coder",
    "qwen3-coder",
    "codellama",
    "codestral",
    "codegemma",
    "deepseek-coder",
    "starcoder",
    "granite-code",
)
STRONG_INSTRUCT_FAMILIES = (
    "qwen3",
    "qwen2.5",
    "llama3.1",
    "llama3.3",
    "gemma3",
    "mistral",
    "mistral-nemo",
    "phi4",
    "phi3.5",
    "deepseek",
)
# Base (non-instruct) checkpoints rarely follow the tool protocol.
BASE_MODEL_MARKERS = ("-base", ":base", "text-")

# VRAM fit thresholds (total params, billions) for a 12 GB card at ~Q4_K_M.
FITS_COMFORTABLY_MAX_B = 9.0   # full 16k context, snappy
FITS_TIGHT_MAX_B = 15.0        # fits at Q4, trim context if OOM
# Above FITS_TIGHT_MAX_B the model spills to system RAM on a 3060 12 GB.

# Reliability thresholds (effective/active params, billions).
RELIABLE_MIN_B = 7.0           # high reliability for tool loops
USABLE_MIN_B = 4.0             # medium; below this is best-effort


@dataclass
class ModelProfile:
    name: str
    params_b: float | None          # total params (drives VRAM fit)
    active_b: float | None          # active params for MoE (drives reliability)
    family: str
    is_coder: bool
    is_base: bool
    fit: str                        # comfortable | tight | exceeds | unknown
    reliability: str                # high | medium | low | unknown
    tier: str                       # recommended | supported | best_effort | unsupported
    notes: list[str] = field(default_factory=list)

    # Generation defaults the rest of the app reads off the profile.
    @property
    def use_few_shot(self) -> bool:
        """Weaker models benefit most from worked tool-call examples."""
        return self.reliability in {"low", "medium", "unknown"}

    @property
    def trim_prompt(self) -> bool:
        """Tiny-context / low-reliability models do better with a lean prompt."""
        return self.reliability == "low"

    @property
    def suggested_num_ctx(self) -> int:
        # Very small models usually ship short native contexts; don't over-ask.
        if self.params_b is not None and self.params_b < USABLE_MIN_B:
            return min(DEFAULT_NUM_CTX, 8192)
        return DEFAULT_NUM_CTX

    @property
    def meets_standard(self) -> bool:
        return self.tier in {"recommended", "supported"}


def _parse_params(name: str):
    """Return (total_b, active_b) parsed from an Ollama tag.

    Handles plain sizes (``7b``, ``14b``, ``1.5b``) and MoE tags that encode
    active params separately (``30b-a3b`` -> total 30, active 3).
    """
    lowered = name.lower()
    moe = re.search(r"(\d+(?:\.\d+)?)\s*b[-_]a(\d+(?:\.\d+)?)\s*b", lowered)
    if moe:
        return float(moe.group(1)), float(moe.group(2))
    sizes = [float(m) for m in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b\b", lowered)]
    if sizes:
        total = max(sizes)
        return total, total
    return None, None


def _detect_family(name: str) -> tuple[str, bool]:
    lowered = name.lower()
    for fam in CODER_FAMILIES:
        if fam in lowered:
            return fam, True
    for fam in STRONG_INSTRUCT_FAMILIES:
        if fam in lowered:
            return fam, False
    base = lowered.split(":", 1)[0].split("/", 1)[-1]
    return base, False


def classify_model(name: str) -> ModelProfile:
    """Classify a model tag into a capability profile for the 3060 12 GB target."""
    name = (name or "").strip()
    total_b, active_b = _parse_params(name)
    family, is_coder = _detect_family(name)
    is_base = any(marker in name.lower() for marker in BASE_MODEL_MARKERS)
    notes: list[str] = []

    # --- VRAM fit (total params) ---
    if total_b is None:
        fit = "unknown"
        notes.append("Could not infer size from the tag; assuming it fits but verify VRAM.")
    elif total_b <= FITS_COMFORTABLY_MAX_B:
        fit = "comfortable"
    elif total_b <= FITS_TIGHT_MAX_B:
        fit = "tight"
        notes.append(f"~{total_b:g}B fits a 12 GB card at Q4_K_M but with limited context headroom; lower LOCAL_CODE_NUM_CTX if it OOMs.")
    else:
        fit = "exceeds"
        notes.append(f"~{total_b:g}B exceeds 12 GB VRAM at a usable quant; it will offload to CPU and run slowly on a 3060.")

    # --- Reliability for the tool loop (active params + family) ---
    effective_b = active_b if active_b is not None else total_b
    if is_base:
        reliability = "low"
        notes.append("Base (non-instruct) checkpoint; unlikely to follow the tool protocol. Prefer an instruct/coder variant.")
    elif effective_b is None:
        reliability = "unknown"
    elif effective_b >= RELIABLE_MIN_B:
        reliability = "high"
    elif effective_b >= USABLE_MIN_B:
        reliability = "medium"
        if not is_coder:
            notes.append("4-7B non-coder model: fine for chat and inspection, occasionally flaky on multi-file edits.")
    else:
        reliability = "low"
        notes.append("Under ~4B: good for chat and single-file reads, unreliable for multi-step edits.")

    if active_b is not None and total_b is not None and active_b < total_b:
        notes.append(f"MoE: ~{total_b:g}B total / ~{active_b:g}B active. Reliability tracks active params; VRAM tracks total.")

    # --- Combine into a tier ---
    if fit == "exceeds":
        tier = "unsupported"
    elif reliability == "low":
        tier = "best_effort"
    elif reliability == "high" and fit == "comfortable":
        tier = "recommended"
    elif reliability in {"high", "medium"}:
        tier = "supported"
    else:  # unknown reliability that at least fits
        tier = "supported"

    return ModelProfile(
        name=name,
        params_b=total_b,
        active_b=active_b,
        family=family,
        is_coder=is_coder,
        is_base=is_base,
        fit=fit,
        reliability=reliability,
        tier=tier,
        notes=notes,
    )


TIER_LABELS = {
    "recommended": "recommended",
    "supported": "supported",
    "best_effort": "best-effort (below standard)",
    "unsupported": "unsupported on 12 GB",
}


def advisory_lines(frontend_model: str, backend_model: str) -> list[str]:
    """Human-readable startup advisory about the configured models.

    Also warns about the dual-model VRAM trap: two distinct models whose
    combined footprint exceeds the card force a reload on every role switch.
    """
    lines: list[str] = []
    seen = []
    for role, model in (("frontend", frontend_model), ("backend", backend_model)):
        if model in [m for _, m in seen]:
            seen.append((role, model))
            continue
        seen.append((role, model))
        profile = classify_model(model)
        lines.append(f"{role}: {model} — {TIER_LABELS.get(profile.tier, profile.tier)}")
        for note in profile.notes:
            lines.append(f"    · {note}")

    if frontend_model != backend_model:
        fp = classify_model(frontend_model)
        bp = classify_model(backend_model)
        combined = (fp.params_b or 0) + (bp.params_b or 0)
        if combined > FITS_TIGHT_MAX_B:
            lines.append(
                f"    · Two distinct models (~{combined:g}B combined) won't stay resident together on 12 GB; "
                "every frontend↔backend switch reloads weights. Use one shared model, or a small frontend "
                "(e.g. qwen3:4b) with a 7B backend so both fit."
            )
    return lines


# The published standard, surfaced in docs and the --models-help output.
RECOMMENDED_STANDARD = "qwen2.5-coder:7b"
RECOMMENDED_CEILING = "qwen2.5-coder:14b"
BEST_EFFORT_FLOOR = "qwen3:4b"
