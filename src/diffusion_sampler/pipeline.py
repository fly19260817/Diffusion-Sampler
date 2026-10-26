"""Model-aware loading, component composition, and unified sampling."""
import inspect
import json
from pathlib import Path

import diffusers
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from .scheduler import FlowScheduler
from .solver import PredictionType, Solver, SolverProtocol


class ComponentSource:
    """Load one component with its actual Diffusers/Transformers class.

    kwargs may contain subfolder, revision, torch_dtype/dtype, quantization_config,
    local_files_only, etc. Tokenizers must be paired with compatible encoders.
    """

    def __init__(self, component_class, model_id, kwargs=None):
        self.component_class = component_class
        self.model_id = model_id
        self.kwargs = dict(kwargs or {})

    def load(self):
        return self.component_class.from_pretrained(str(self.model_id), **self.kwargs)


class SamplingResult:
    def __init__(self, images, raw):
        self.images = images
        self.raw = raw


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
    name = family.lower()
    return _ALIASES.get(name, name)


def _load_model_index(model_id, load_kwargs):
    """Prefer the modular index, falling back only when that file is absent."""
    directory = Path(model_id)
    hub_options = {}
    for key in ("revision", "cache_dir", "token", "local_files_only"):
        if key in load_kwargs:
            hub_options[key] = load_kwargs[key]

    for filename in ("modular_model_index.json", "model_index.json"):
        if directory.is_dir():
            candidate = directory / filename
            if not candidate.is_file():
                continue
        else:
            try:
                candidate = Path(hf_hub_download(
                    repo_id=model_id, filename=filename,
                    **hub_options,
                ))
            except EntryNotFoundError as error:
                if isinstance(error, LocalEntryNotFoundError) and not hub_options.get("local_files_only"):
                    raise
                # Classic repositories do not have a modular index.
                # Authentication, network and malformed-config errors propagate.
                continue
        return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No Diffusers model index found in {model_id}")


def _detect_family(model_id, load_kwargs):
    config = _load_model_index(model_id, load_kwargs)
    name = config["_class_name"]
    if name.startswith("Flux3"):
        return "flux3"
    supported = {
        "StableDiffusion3Pipeline": "sd35", "FluxPipeline": "flux1",
        "QwenImagePipeline": "qwen-image",
    }
    return supported[name]


def _load_components(components):
    loaded = {}
    for name, component in components.items():
        if isinstance(component, ComponentSource):
            component = component.load()
        loaded[name] = component
    return loaded


def _check_component_names(components, accepted_names):
    # Diffusers may silently ignore unknown names, so check at this boundary.
    unknown = set(components) - set(accepted_names)
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")


def _load_classic_pipeline(backend_class, model_id, components, dtype, load_kwargs):
    accepted = inspect.signature(backend_class.__init__).parameters
    _check_component_names(components, accepted)
    loaded = _load_components(components)
    options = dict(load_kwargs)
    if dtype is not None:
        options["torch_dtype"] = dtype
    return backend_class.from_pretrained(model_id, **loaded, **options)


def _load_modular_pipeline(
    backend_class, model_id, components, dtype, load_kwargs, component_load_kwargs,
):
    backend = backend_class.from_pretrained(model_id, **load_kwargs)
    _check_component_names(components, backend.components)
    loaded = _load_components(components)
    backend.update_components(**loaded)

    remaining = []
    for name in backend.pretrained_component_names:
        # Explicit None disables an encoder; do not reload it.
        if name not in loaded and getattr(backend, name) is None:
            remaining.append(name)

    options = {}
    for key in ("cache_dir", "token", "local_files_only"):
        if key in load_kwargs:
            options[key] = load_kwargs[key]
    options.update(component_load_kwargs or {})
    if dtype is not None and "dtype" not in options:
        options["dtype"] = {"vae": torch.float32, "default": dtype}
    backend.load_components(names=remaining, **options)
    return backend


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
        components: dict | None = None,
        scheduler=None, solver: SolverProtocol | None = None,
        torch_dtype: torch.dtype | None = None,
        component_load_kwargs: dict | None = None,
        **load_kwargs,
    ):
        """Components can be loaded objects, ComponentSource objects, or None.

        None explicitly disables a component (e.g. an encoder with precomputed
        embeddings). Omitted components come from the model repository.
        """
        model_id = str(model_id)
        if family == "auto":
            family = _detect_family(model_id, load_kwargs)
        else:
            family = _resolve_family(family)
        class_name, modular = _FAMILIES[family]
        backend_class = getattr(diffusers, class_name)
        supplied = dict(components or {})
        if "scheduler" in supplied:
            raise ValueError("Pass scheduler through the dedicated scheduler= argument")
        if not modular and component_load_kwargs:
            raise ValueError("component_load_kwargs is for modular pipelines; use ComponentSource for others")
        if modular:
            backend = _load_modular_pipeline(
                backend_class, model_id, supplied, torch_dtype,
                load_kwargs, component_load_kwargs,
            )
        else:
            backend = _load_classic_pipeline(
                backend_class, model_id, supplied, torch_dtype, load_kwargs,
            )
        result = cls(backend, modular=modular)
        result.configure_sampling(scheduler=scheduler, solver=solver)
        return result

    @classmethod
    def from_components(cls, *, family: str, scheduler, blocks=None, **components):
        """Build entirely from supplied components; FLUX.3 accepts custom blocks."""
        class_name, modular = _FAMILIES[_resolve_family(family)]
        loaded = _load_components(components)
        if modular:
            if blocks is None:
                blocks = diffusers.Flux3AutoBlocks()
            backend = blocks.init_pipeline()
            _check_component_names(loaded, backend.components)
            backend.update_components(scheduler=scheduler, **loaded)
        else:
            if blocks is not None:
                raise ValueError("blocks is only supported for modular pipelines")
            backend = getattr(diffusers, class_name)(scheduler=scheduler, **loaded)
        return cls(backend, modular=modular)

    @classmethod
    def from_backend(cls, backend, *, solver: SolverProtocol | None = None):
        modular = isinstance(backend, diffusers.ModularPipeline)
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
        _check_component_names(components, self.components)
        if "scheduler" in components:
            raise ValueError("Use configure_sampling() for scheduler changes")
        loaded = _load_components(components)
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
        if isinstance(raw, tuple):
            images = raw[0]
        else:
            images = raw.images
        return SamplingResult(images=images, raw=raw)
