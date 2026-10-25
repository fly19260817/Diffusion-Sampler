"""Model-aware loading, component composition, and unified sampling."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import diffusers
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .scheduler import FlowScheduler
from .solver import PredictionType, Solver, SolverProtocol


@dataclass(frozen=True)
class ComponentSource:
    """Load one component with its actual Diffusers/Transformers class.

    kwargs may contain subfolder, revision, torch_dtype/dtype, quantization_config,
    local_files_only, etc. Tokenizers must be paired with compatible encoders.
    """

    component_class: type
    model_id: str | Path
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def load(self):
        return self.component_class.from_pretrained(str(self.model_id), **dict(self.kwargs))


@dataclass
class SamplingResult:
    images: Any
    raw: Any


# Concrete classes are resolved lazily so SD3/FLUX1/Qwen also work without FLUX3.
_FAMILIES = {
    "sd35": ("StableDiffusion3Pipeline", False),
    "flux1": ("FluxPipeline", False),
    "flux3": ("ModularPipeline", True),
    "qwen-image": ("QwenImagePipeline", False),
}
_ALIASES = {
    "sd3": "sd35", "sd3.5": "sd35", "stable-diffusion-3.5": "sd35",
    "flux": "flux1", "flux.1": "flux1", "flux.3": "flux3",
    "qwen": "qwen-image", "qwen_image": "qwen-image",
}


def _resolve_family(family: str) -> str:
    name = _ALIASES.get(family.lower(), family.lower())
    if name not in _FAMILIES:
        raise ValueError(f"Unknown model family {family!r}; choose {list(_FAMILIES)}")
    return name


def _detect_family(model_id: str, load_kwargs: Mapping[str, Any]) -> str:
    directory = Path(model_id)
    config = None
    for filename in ("modular_model_index.json", "model_index.json"):
        if directory.is_dir():
            candidate = directory / filename
            if not candidate.is_file():
                continue
        else:
            try:
                candidate = Path(hf_hub_download(
                    repo_id=model_id, filename=filename,
                    **{key: load_kwargs[key] for key in (
                        "revision", "cache_dir", "token", "local_files_only",
                    ) if key in load_kwargs},
                ))
            except EntryNotFoundError:
                continue
        config = json.loads(candidate.read_text(encoding="utf-8"))
        break
    if config is None:
        raise ValueError("No model index found; supply a Diffusers model directory/repository")
    name = str(config.get("_class_name", ""))
    if name.startswith("Flux3"):
        return "flux3"
    supported = {
        "StableDiffusion3Pipeline": "sd35", "FluxPipeline": "flux1",
        "QwenImagePipeline": "qwen-image",
    }
    if name not in supported:
        raise ValueError(
            f"Cannot infer a supported text-to-image family from {name!r}; "
            "pass family explicitly or use from_backend()"
        )
    return supported[name]


class SamplingPipeline:
    """Compose official model pipelines while retaining their native defaults.

    The native backend remains available for LoRA, callbacks, quantization and
    family-specific features. Instances/schedulers are not thread-safe.
    """

    def __init__(self, backend, *, modular: bool = False):
        self.backend = backend
        self.modular = modular
        self._default_scheduler = backend.scheduler

    @classmethod
    def from_pretrained(
        cls, model_id: str | Path, *, family: str = "auto",
        components: Mapping[str, Any] | None = None,
        scheduler=None, solver: SolverProtocol | None = None,
        torch_dtype: torch.dtype | None = None,
        component_load_kwargs: Mapping[str, Any] | None = None,
        **load_kwargs,
    ):
        """Components can be loaded objects, ComponentSource objects, or None.

        None explicitly disables a component (e.g. an encoder with precomputed
        embeddings). Omitted components come from the model repository.
        """
        model_id = str(model_id)
        family = _detect_family(model_id, load_kwargs) if family == "auto" else _resolve_family(family)
        class_name, modular = _FAMILIES[family]
        backend_class = getattr(diffusers, class_name, None)
        if backend_class is None:
            raise ImportError(f"{class_name} is missing; install a Diffusers build supporting {family}")
        if family == "flux3" and not hasattr(diffusers, "Flux3AutoBlocks"):
            raise ImportError("FLUX.3 requires the source Diffusers dependency in environments/")
        supplied = dict(components or {})
        if "scheduler" in supplied:
            raise ValueError("Pass scheduler through the dedicated scheduler= argument")
        duplicated = set(supplied) & set(load_kwargs)
        if duplicated:
            raise ValueError(f"Components supplied twice: {sorted(duplicated)}")
        if not modular and component_load_kwargs:
            raise ValueError("component_load_kwargs is for modular pipelines; use ComponentSource for others")
        if modular:
            backend = backend_class.from_pretrained(model_id, **load_kwargs)
            unknown = set(supplied) - set(backend.components)
            if unknown:
                raise ValueError(f"Unknown components: {sorted(unknown)}")
            loaded = {
                name: value.load() if isinstance(value, ComponentSource) else value
                for name, value in supplied.items()
            }
            if loaded:
                backend.update_components(**loaded)
            # Exclude explicit None entries: load_components() would otherwise reload them.
            remaining = [
                name for name in backend.pretrained_component_names
                if name not in loaded and getattr(backend, name, None) is None
            ]
            options = {
                key: load_kwargs[key] for key in ("cache_dir", "token", "local_files_only")
                if key in load_kwargs
            }
            options.update(dict(component_load_kwargs or {}))
            if torch_dtype is not None and "dtype" not in options:
                options["dtype"] = {"vae": torch.float32, "default": torch_dtype}
            backend.load_components(names=remaining, **options)
        else:
            import inspect
            accepted = set(inspect.signature(backend_class.__init__).parameters)
            unknown = set(supplied) - accepted
            if unknown:
                raise ValueError(f"Unknown components: {sorted(unknown)}")
            loaded = {
                name: value.load() if isinstance(value, ComponentSource) else value
                for name, value in supplied.items()
            }
            options = dict(load_kwargs)
            if torch_dtype is not None:
                options["torch_dtype"] = torch_dtype
            backend = backend_class.from_pretrained(model_id, **loaded, **options)
        result = cls(backend, modular=modular)
        result.configure_sampling(scheduler=scheduler, solver=solver)
        return result

    @classmethod
    def from_components(cls, *, family: str, scheduler, blocks=None, **components):
        """Build entirely from supplied components; FLUX.3 accepts custom blocks."""
        class_name, modular = _FAMILIES[_resolve_family(family)]
        loaded = {
            name: value.load() if isinstance(value, ComponentSource) else value
            for name, value in components.items()
        }
        if modular:
            if blocks is None:
                blocks_class = getattr(diffusers, "Flux3AutoBlocks", None)
                if blocks_class is None:
                    raise ImportError("Install the FLUX.3-capable Diffusers source dependency")
                blocks = blocks_class()
            backend = blocks.init_pipeline()
            unknown = set(loaded) - set(backend.components)
            if unknown:
                raise ValueError(f"Unknown components: {sorted(unknown)}")
            backend.update_components(scheduler=scheduler, **loaded)
        else:
            if blocks is not None:
                raise ValueError("blocks is only supported for modular pipelines")
            backend = getattr(diffusers, class_name)(scheduler=scheduler, **loaded)
        return cls(backend, modular=modular)

    @classmethod
    def from_backend(cls, backend, *, solver: SolverProtocol | None = None):
        modular = isinstance(backend, diffusers.ModularPipeline) if hasattr(diffusers, "ModularPipeline") else False
        result = cls(backend, modular=modular)
        result.configure_sampling(solver=solver)
        return result

    @property
    def components(self):
        return dict(self.backend.components)

    @property
    def scheduler(self):
        return self.backend.scheduler

    def replace_components(self, **components):
        """Replace components before configuring device placement/offloading."""
        unknown = set(components) - set(self.components)
        if unknown:
            raise ValueError(f"Unknown components: {sorted(unknown)}")
        if "scheduler" in components:
            raise ValueError("Use configure_sampling() for scheduler changes")
        loaded = {
            name: value.load() if isinstance(value, ComponentSource) else value
            for name, value in components.items()
        }
        if self.modular:
            self.backend.update_components(**loaded)
        else:
            self.backend.register_modules(**loaded)
        return self

    def configure_sampling(self, *, scheduler=None, solver: SolverProtocol | None = None):
        """None restores the loaded model's original scheduler/integrator.

        An explicit scheduler supplies both its grid and native numerical method.
        An explicit solver overrides the method and retains that scheduler's grid.
        Independent solvers currently require a FlowMatchEuler-compatible grid.
        """
        selected = self._default_scheduler if scheduler is None else scheduler
        if solver is not None:
            if isinstance(solver, Solver) and solver.prediction_type != PredictionType.FLOW:
                raise ValueError(
                    "These model adapters emit flow predictions. Use flow_prediction here; "
                    "other parameterizations are available with Solver/FlowScheduler directly "
                    "for models that actually emit those outputs."
                )
            if "Distilled" in type(self.backend).__name__ and (
                not isinstance(solver, Solver) or solver.method != "euler"
            ):
                raise ValueError("Step-distilled FLUX.3 requires its native method or flow Euler")
            selected = FlowScheduler.from_scheduler(selected, solver=solver)
        if self.modular:
            self.backend.update_components(scheduler=selected)
        else:
            self.backend.register_modules(scheduler=selected)
        return self

    def to(self, device, **kwargs):
        self.backend.to(device, **kwargs)
        return self

    def enable_cpu_offload(self, *, device: str = "cuda", gpu_id: int = 0):
        """Classic model offload; modular pipelines use leaf-level group offload."""
        if self.modular:
            from diffusers.hooks import apply_group_offloading
            for component in self.components.values():
                if isinstance(component, torch.nn.Module):
                    apply_group_offloading(
                        component, onload_device=torch.device(f"{device}:{gpu_id}"),
                        offload_device=torch.device("cpu"),
                        offload_type="leaf_level", use_stream=False,
                    )
        else:
            self.backend.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        return self

    @torch.inference_mode()
    def __call__(self, prompt=None, *, seed: int | None = None, **kwargs):
        """Pass model-specific arguments through; omitted sampling options stay native."""
        if seed is not None:
            if kwargs.get("generator") is not None:
                raise ValueError("Pass seed or generator, not both")
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        if prompt is not None:
            kwargs["prompt"] = prompt
        if self.modular:
            if "return_dict" in kwargs or "output" in kwargs:
                raise ValueError("Use backend(...) for raw modular output selection")
            if kwargs.get("output_type") == "latent":
                kwargs.pop("output_type")
                output = "latents"
            else:
                output = "images"
            raw = self.backend(output=output, **kwargs)
            return SamplingResult(images=raw, raw=raw)
        raw = self.backend(**kwargs)
        images = raw.images if hasattr(raw, "images") else raw[0]
        return SamplingResult(images=images, raw=raw)
