"""Diffusers-compatible time-grid scheduler with a replaceable integrator."""
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

    # Keep the inherited constructor intact for Diffusers config loading.
    solver = None

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
            if self.config.get("invert_sigmas", False):
                raise ValueError("Custom solvers require descending rectified-flow sigmas")
            if self.config.get("stochastic_sampling", False):
                raise ValueError("Custom deterministic solvers cannot use stochastic_sampling")
            solver.reset()
        self.solver = solver
        return self

    def set_timesteps(
        self, num_inference_steps=None, device=None, sigmas=None, mu=None,
        timesteps=None,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps, device=device,
            sigmas=sigmas, mu=mu, timesteps=timesteps,
        )
        if self.solver is not None:
            grid = self.sigmas
            if (
                grid.ndim != 1
                or grid.numel() < 2
                or not torch.isfinite(grid).all()
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
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps() before step()")
        if self.step_index is None:
            self._init_step_index(timestep)
        index = self.step_index
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
