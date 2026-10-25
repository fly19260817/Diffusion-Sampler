"""Diffusers-compatible time-grid scheduler with a replaceable integrator."""
from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)

from .solver import SolverProtocol


class FlowScheduler(FlowMatchEulerDiscreteScheduler):
    """Reuse model-specific shifting/grid logic and optionally replace only step().

    Inherits the upstream constructor/config serialization unchanged. Use
    FlowScheduler.from_config(model.scheduler.config) to preserve model settings,
    then set_solver(). Without an explicit solver, upstream Euler behavior remains.
    Runtime solver objects/history are deliberately not serialized.
    """

    @classmethod
    def from_scheduler(cls, scheduler, *, solver: SolverProtocol | None = None):
        if type(scheduler) not in {FlowMatchEulerDiscreteScheduler, cls}:
            raise TypeError(
                "Independent solvers require FlowMatchEulerDiscreteScheduler or "
                "FlowScheduler; an arbitrary scheduler cannot be converted safely"
            )
        result = cls.from_config(scheduler.config)
        result.set_solver(solver)
        return result

    def set_solver(self, solver: SolverProtocol | None):
        if solver is not None:
            if not isinstance(solver, SolverProtocol):
                raise TypeError("solver must implement reset() and step()")
            if self.config.get("invert_sigmas", False):
                raise ValueError("Custom solvers require descending rectified-flow sigmas")
            if self.config.get("stochastic_sampling", False):
                raise ValueError("Custom deterministic solvers cannot use stochastic_sampling")
            solver.reset()
        self._solver = solver
        return self

    @property
    def solver(self):
        return getattr(self, "_solver", None)

    def set_timesteps(
        self, num_inference_steps=None, device=None, sigmas=None, mu=None,
        timesteps=None,
    ):
        if num_inference_steps is not None and (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, int)
            or num_inference_steps < 1
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        for name, values in (("sigmas", sigmas), ("timesteps", timesteps)):
            if values is not None:
                grid = torch.as_tensor(values, dtype=torch.float64)
                if grid.ndim != 1 or not grid.numel() or not torch.isfinite(grid).all():
                    raise ValueError(f"{name} must be a nonempty finite 1-D sequence")
                if not (grid[:-1] > grid[1:]).all():
                    raise ValueError(f"{name} must be strictly descending")
        super().set_timesteps(
            num_inference_steps=num_inference_steps, device=device,
            sigmas=sigmas, mu=mu, timesteps=timesteps,
        )
        if self.solver is not None:
            grid = self.sigmas
            if (
                not torch.isfinite(grid).all()
                or not ((grid >= 0) & (grid <= 1)).all()
                or not (grid[:-1] > grid[1:]).all()
            ):
                raise ValueError("Custom solver requires a strictly descending sigma grid in [0,1]")
            self.solver.reset()

    def set_begin_index(self, begin_index: int = 0):
        if not 0 <= begin_index < len(self.timesteps):
            raise ValueError("begin_index is outside the timestep grid")
        super().set_begin_index(begin_index)
        self._step_index = None
        if self.solver is not None:
            self.solver.reset()

    def step(
        self, model_output, timestep, sample,
        s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
        generator=None, per_token_timesteps=None, return_dict=True,
    ):
        if self.solver is None:
            return super().step(
                model_output, timestep, sample, s_churn=s_churn,
                s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
                generator=generator, per_token_timesteps=per_token_timesteps,
                return_dict=return_dict,
            )
        if per_token_timesteps is not None:
            raise NotImplementedError("Custom solvers currently require a shared scalar timestep")
        if s_churn != 0 or s_noise != 1 or s_tmin != 0 or s_tmax != float("inf"):
            raise ValueError("Stochastic/churn parameters are unsupported by custom solvers")
        t = torch.as_tensor(timestep)
        if t.numel() != 1 or not t.is_floating_point():
            raise ValueError("Pass a floating-point scheduler timestep, not a loop index")
        if getattr(self, "num_inference_steps", None) is None:
            raise RuntimeError("Call set_timesteps() before step()")
        if self.step_index is None:
            self._init_step_index(timestep)
        index = self.step_index
        if index >= len(self.timesteps):
            raise RuntimeError("Schedule exhausted; call set_timesteps() before another request")
        expected = self.timesteps[index]
        if not torch.isclose(t.to(expected), expected, rtol=1e-5, atol=1e-5).all():
            raise ValueError("Timestep does not match scheduler position")
        previous = self.solver.step(
            model_output, sample,
            float(self.sigmas[index]), float(self.sigmas[index + 1]),
        )
        self._step_index += 1
        if return_dict:
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=previous)
        return (previous,)
