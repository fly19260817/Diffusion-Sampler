"""Numerical integration for the rectified-flow path x(s)=(1-s)x0+s*eps."""
from enum import Enum

import torch


class PredictionType(str, Enum):
    FLOW = "flow_prediction"  # eps - x0 = dx/dsigma
    EPSILON = "epsilon"
    SAMPLE = "sample"  # x0
    V = "v_prediction"  # (1-sigma)*eps - sigma*x0, NOT a VP noise path


class SolverProtocol:
    """Optional base class for solvers; implementing these two methods is enough."""

    def reset(self) -> None:
        raise NotImplementedError

    def step(
        self, model_output: torch.Tensor, sample: torch.Tensor,
        sigma: float, sigma_next: float,
    ) -> torch.Tensor:
        raise NotImplementedError


class Solver(SolverProtocol):
    """Euler or variable-step AB2, with an explicit model-output parameterization.

    All parameterizations describe the SAME linear rectified-flow path.
    v_prediction here does not turn this into a VP/DDPM solver.
    Stateful instances must not be shared across concurrent requests.
    """

    def __init__(
        self, method: str = "euler",
        prediction_type: str | PredictionType = PredictionType.FLOW,
    ):
        if method not in {"euler", "ab2"}:
            raise ValueError("method must be 'euler' or 'ab2'")
        self.method = method
        self.prediction_type = PredictionType(prediction_type)
        self.reset()

    def reset(self) -> None:
        self._previous_derivative = None
        self._previous_step = None
        self._expected_sigma = None

    def to_derivative(
        self, model_output: torch.Tensor, sample: torch.Tensor, sigma: float,
    ) -> torch.Tensor:
        if model_output.shape != sample.shape:
            raise ValueError("model_output and sample must have identical shapes")
        if not 0.0 <= sigma <= 1.0:
            raise ValueError("Rectified-flow sigma must be finite and in [0, 1]")
        # Accumulate in float32, preserving float64 for numerical research.
        dtype = torch.float64 if sample.dtype == torch.float64 else torch.float32
        prediction = model_output.to(dtype)
        if self.prediction_type == PredictionType.FLOW:
            return prediction
        x = sample.to(dtype)
        alpha = 1.0 - sigma
        if self.prediction_type == PredictionType.EPSILON:
            if alpha == 0.0:
                raise ValueError("epsilon prediction is singular at sigma=1; start below 1")
            return (prediction - x) / alpha
        if self.prediction_type == PredictionType.SAMPLE:
            if sigma == 0.0:
                raise ValueError("x0 prediction is singular at sigma=0")
            return (x - prediction) / sigma
        # Solve x=alpha*x0+sigma*eps, v=alpha*eps-sigma*x0.
        return ((sigma - alpha) * x + prediction) / (alpha * alpha + sigma * sigma)

    @torch.no_grad()
    def step(
        self, model_output: torch.Tensor, sample: torch.Tensor,
        sigma: float, sigma_next: float,
    ) -> torch.Tensor:
        sigma = float(sigma)
        sigma_next = float(sigma_next)
        if not 0.0 <= sigma_next < sigma <= 1.0:
            raise ValueError("Require 0 <= sigma_next < sigma <= 1")
        if not sample.is_floating_point() or not model_output.is_floating_point():
            raise TypeError("Solver inputs must be floating-point tensors")
        if self._expected_sigma is not None and sigma != self._expected_sigma:
            raise ValueError("Discontinuous trajectory; call reset() before a new sample")
        derivative = self.to_derivative(model_output, sample, sigma)
        step_size = sigma_next - sigma
        update = derivative
        if self.method == "ab2" and self._previous_derivative is not None:
            previous = self._previous_derivative
            if previous.shape != derivative.shape:
                raise ValueError("Sample shape changed; reset the solver")
            # AB2 integrates the line through the current and previous derivatives.
            ratio = step_size / (2.0 * self._previous_step)
            update = (1.0 + ratio) * derivative - ratio * previous
        result = (sample.to(derivative.dtype) + step_size * update).to(sample.dtype)
        if self.method == "ab2":
            self._previous_derivative = derivative.detach().clone()
        self._previous_step = step_size
        self._expected_sigma = sigma_next
        return result
