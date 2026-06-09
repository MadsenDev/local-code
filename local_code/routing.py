"""Resolve requested single/adaptive/dual model execution modes."""

from .hardware import recommend_routing


def resolve_model_routing(requested, provider_is_local, hardware, frontend_model, backend_model, num_ctx=16384):
    if requested == "single":
        return backend_model, backend_model, {"requested": requested, "mode": "single", "reason": "Single-model mode was explicitly requested; the coder/backend model is shared by both roles."}
    if requested == "dual":
        return frontend_model, backend_model, {"requested": requested, "mode": "dual", "reason": "Dual-model mode was explicitly requested."}
    if not provider_is_local:
        mode = "single" if frontend_model == backend_model else "dual"
        return frontend_model, backend_model, {"requested": requested, "mode": mode, "reason": "Cloud providers do not pay local model residency costs."}
    decision = recommend_routing(hardware, frontend_model, backend_model, num_ctx)
    if decision["mode"] == "single":
        # Prefer the coder/backend model as the one shared model: capability is
        # more important than a small conversational quality difference.
        return backend_model, backend_model, {"requested": requested, **decision}
    return frontend_model, backend_model, {"requested": requested, **decision}
