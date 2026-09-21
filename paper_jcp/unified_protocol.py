"""Core components for the unified JCP small-data protocol.

This module deliberately does not execute experiments.  It provides the
normalised common PINN backbone, vanilla/H/HO loss balancing, nested labelled
designs, time interpolation/extrapolation splits, and per-epoch residual
resampling used by :mod:`paper_jcp.run_unified_protocol`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from tqdm import trange
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from benchmarking.pinn_ablation_utils import (
    compute_multioutput_metrics,
    compute_scalar_metrics,
)
from loss_balancong_algorithms.algos import HierarchicalOrthogonalPINNLossBalancer


LABEL_BUDGETS = (20, 50, 100, 200, 500, 1000)
TARGET_NAMES = ("Pressure", "Soil", "Swat")
TEST_NAMES = ("test1", "test2", "test3", "test4")

DOMAIN_BOUNDS = {
    "test1": ([0.0, 0.0], [0.05, 1.0]),
    "test2": ([0.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
    "test3": ([0.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
    "test4": (
        [0.0, 0.0, 0.0, 1.5, 3.0, 0.9, 0.2],
        [3.0, 1.0, 1.0, 2.5, 5.0, 1.1, 0.4],
    ),
}


def heat_exact_solution(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Analytical solution used by the self-contained Test 1 benchmark."""

    return (
        2.0
        + torch.exp(-4.0 * torch.pi**2 * t) * torch.sin(2.0 * torch.pi * x)
        + torch.exp(-16.0 * torch.pi**2 * t) * torch.cos(4.0 * torch.pi * x)
    )


class _HeatEquationModule:
    """Compatibility object for the shared heat-loss implementation."""

    thermal_conductivity_equation = staticmethod(heat_exact_solution)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    depth: int = 6
    activation: str = "tanh"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50_000
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    l0_data: float = 1e-2
    l0_ibc: float = 1e-3
    orthogonal_floor: float = 0.5
    collocation_ratio: float = 1.0
    residual_domain: str = "full"
    gradient_clip: float = 1.0
    checkpoint_criterion: str = "balanced"
    log_every: int = 100
    physics_validation_size: int = 512
    device: str = "cpu"
    deterministic: bool = True


@dataclass
class UnifiedBundle:
    test_name: str
    module: object
    data_source: str
    dataset_path: str | None
    time_cutoff: float
    full_time: float
    input_lower: tuple[float, ...]
    input_upper: tuple[float, ...]
    porosity_value: float
    water_viscosity: float
    oil_viscosity: float
    features: np.ndarray | None
    targets: np.ndarray | None
    permeability: np.ndarray | None
    train_candidates: np.ndarray | None
    residual_candidates_full: np.ndarray | None
    residual_candidates_label: np.ndarray | None
    interpolation_x: np.ndarray
    interpolation_y: np.ndarray
    extrapolation_x: np.ndarray
    extrapolation_y: np.ndarray


def resolve_device(requested: str) -> torch.device:
    """Resolve ``cpu``, ``cuda[:index]``, ``mps``, or ``auto`` explicitly."""

    name = requested.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but it is unavailable. Use --device cpu instead."
        )
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def configure_torch_runtime(
    seed: int,
    device: torch.device,
    deterministic: bool = True,
) -> None:
    """Configure repeatable PyTorch execution before model construction."""

    if deterministic:
        # Required by deterministic CUDA GEMM kernels on supported CUDA builds.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "tanh":
        return torch.tanh
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


class UnifiedPINN(nn.Module):
    """One gated residual backbone shared by all four benchmarks.

    Physical inputs are normalised inside ``forward`` from known domain bounds,
    rather than from the sparse labels.  Keeping this affine map inside the
    graph makes automatic differentiation apply the required chain rule.
    """

    def __init__(
        self,
        test_name: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        depth: int = 6,
        activation: str = "tanh",
        input_lower: tuple[float, ...] | list[float] | None = None,
        input_upper: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        if test_name not in DOMAIN_BOUNDS:
            raise ValueError(test_name)
        default_lower, default_upper = DOMAIN_BOUNDS[test_name]
        lower = default_lower if input_lower is None else input_lower
        upper = default_upper if input_upper is None else input_upper
        if len(lower) != input_dim:
            raise ValueError(f"Input dimension mismatch for {test_name}")
        if len(upper) != input_dim:
            raise ValueError(f"Input dimension mismatch for {test_name}")
        self.test_name = test_name
        self.output_dim = output_dim
        self.activation_name = activation
        self.activation = _activation(activation)
        self.register_buffer("input_lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("input_upper", torch.tensor(upper, dtype=torch.float32))

        self.U = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if test_name != "test1":
            self.pressure_scale = nn.Parameter(torch.ones(1))
            self.saturation_scale = nn.Parameter(torch.ones(2))
            self.velocity_scale = nn.Parameter(torch.ones(4))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def normalise(self, physical_input: torch.Tensor) -> torch.Tensor:
        span = torch.clamp(self.input_upper - self.input_lower, min=1e-12)
        return 2.0 * (physical_input - self.input_lower) / span - 1.0

    def forward(self, physical_input: torch.Tensor) -> torch.Tensor:
        z = self.normalise(physical_input)
        h = torch.tanh(self.U(z)) * torch.tanh(self.V(z))
        for index, layer in enumerate(self.hidden_layers):
            candidate = self.activation(layer(h))
            h = h + candidate if index % 2 == 1 else candidate
        raw = self.output_layer(h)
        if self.test_name == "test1":
            return raw

        pressure = torch.sigmoid(raw[:, 0:1] * self.pressure_scale)
        soil = torch.sigmoid(raw[:, 1:2] * self.saturation_scale[0])
        swat = torch.sigmoid(raw[:, 2:3] * self.saturation_scale[1])
        saturation_sum = soil + swat + 1e-8
        soil = soil / saturation_sum
        swat = swat / saturation_sum
        velocities = torch.tanh(raw[:, 3:7]) * self.velocity_scale
        return torch.cat((pressure, soil, swat, velocities), dim=1)


def model_dimensions(test_name: str) -> tuple[int, int]:
    if test_name == "test1":
        return 2, 1
    if test_name in {"test2", "test3"}:
        return 3, 7
    if test_name == "test4":
        return 7, 7
    raise ValueError(test_name)


def build_model(bundle: UnifiedBundle, config: ModelConfig) -> UnifiedPINN:
    input_dim, output_dim = model_dimensions(bundle.test_name)
    return UnifiedPINN(
        test_name=bundle.test_name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        activation=config.activation,
        input_lower=bundle.input_lower,
        input_upper=bundle.input_upper,
    )


def _fixed_split(
    indices: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("Cannot create a test split from an empty set")
    return rng.choice(indices, size=min(size, len(indices)), replace=False)


def _corrected_fixed_flow_arrays(
    test_name: str,
    dataset_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """Load and flatten a corrected Test 2/3 finite-volume solution."""

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Corrected {test_name} dataset not found: {dataset_path}"
        )
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {
            "times",
            "x",
            "y",
            "pressure",
            "water_saturation",
            "oil_saturation",
            "permeability",
            "porosity",
            "metadata_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Corrected dataset {dataset_path} is missing: {sorted(missing)}"
            )
        times = np.asarray(data["times"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        pressure = np.asarray(data["pressure"], dtype=np.float64)
        swat = np.asarray(data["water_saturation"], dtype=np.float64)
        soil = np.asarray(data["oil_saturation"], dtype=np.float64)
        permeability_2d = np.asarray(data["permeability"], dtype=np.float64)
        porosity_2d = np.asarray(data["porosity"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"].item()))

    expected = (len(times), len(x), len(y))
    for name, field in (
        ("pressure", pressure),
        ("water_saturation", swat),
        ("oil_saturation", soil),
    ):
        if field.shape != expected:
            raise ValueError(f"{name} has shape {field.shape}; expected {expected}")
    if permeability_2d.shape != expected[1:]:
        raise ValueError(
            f"permeability has shape {permeability_2d.shape}; expected {expected[1:]}"
        )
    if porosity_2d.shape != expected[1:]:
        raise ValueError(
            f"porosity has shape {porosity_2d.shape}; expected {expected[1:]}"
        )
    if len(times) < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("Corrected times must be strictly increasing")
    if not np.isclose(times[0], 0.0):
        raise ValueError("Corrected dataset must include the exact t=0 state")
    if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
        raise ValueError("Corrected cell-centred coordinates must be increasing")
    if not (np.all((x > 0.0) & (x < 1.0)) and np.all((y > 0.0) & (y < 1.0))):
        raise ValueError(
            f"Corrected {test_name} cell centres must lie inside (0, 1)^2"
        )
    for name, field in (("pressure", pressure), ("Swat", swat), ("Soil", soil)):
        if not np.all(np.isfinite(field)):
            raise ValueError(f"Corrected {name} contains non-finite values")
    if np.any(swat < -1e-10) or np.any(swat > 1.0 + 1e-10):
        raise ValueError("Corrected water saturation lies outside [0, 1]")
    if not np.allclose(soil + swat, 1.0, atol=1e-8):
        raise ValueError("Corrected saturations do not satisfy Soil + Swat = 1")
    if not np.allclose(porosity_2d, porosity_2d.flat[0], atol=1e-12):
        raise ValueError("The current PINN protocol requires constant porosity")

    fluids = metadata.get("fluids", {})
    water_viscosity = float(fluids.get("water_viscosity", 1.0))
    oil_viscosity = float(fluids.get("oil_viscosity", 3.0))
    if water_viscosity <= 0.0 or oil_viscosity <= 0.0:
        raise ValueError("Corrected phase viscosities must be positive")
    relative = metadata.get("relative_permeability", {})
    expected_relative = {
        "water_exponent": 2.0,
        "oil_exponent": 4.0,
        "water_endpoint": 1.0,
        "oil_endpoint": 0.3,
    }
    for name, expected_value in expected_relative.items():
        actual = float(relative.get(name, expected_value))
        if not np.isclose(actual, expected_value):
            raise ValueError(
                f"Corrected {test_name} {name}={actual} does not match PINN {expected_value}"
            )
    boundary = metadata.get("boundary_conditions", {})
    expected_boundary = {
        "left_pressure": 1.0,
        "right_pressure": 0.0,
        "left_inflow_water_fraction": 1.0,
    }
    for name, expected_value in expected_boundary.items():
        actual = float(boundary.get(name, expected_value))
        if not np.isclose(actual, expected_value):
            raise ValueError(
                f"Corrected {test_name} {name}={actual} does not match PINN {expected_value}"
            )
    time_grid, x_grid, y_grid = np.meshgrid(times, x, y, indexing="ij")
    features = np.column_stack(
        (time_grid.reshape(-1), x_grid.reshape(-1), y_grid.reshape(-1))
    ).astype(np.float32)
    targets = np.column_stack(
        (pressure.reshape(-1), soil.reshape(-1), swat.reshape(-1))
    ).astype(np.float32)
    permeability = np.broadcast_to(
        permeability_2d[None, :, :], expected
    ).reshape(-1).astype(np.float32)
    return (
        features,
        targets,
        permeability,
        float(porosity_2d.flat[0]),
        water_viscosity,
        oil_viscosity,
    )


def _sample_unique_integers(
    total: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample integers without allocating an array of length total."""

    if size < 0 or size > total:
        raise ValueError(f"Cannot sample {size} unique values from {total}")
    selected: set[int] = set()
    # Floyd's algorithm is exact and uses O(size) rather than O(total) memory.
    for upper in range(total - size, total):
        candidate = int(rng.integers(0, upper + 1))
        selected.add(upper if candidate in selected else candidate)
    result = np.fromiter(selected, dtype=np.int64, count=size)
    rng.shuffle(result)
    return result


def _test4_codes_for_times(
    *,
    sample_count: int,
    times_per_sample: int,
    nx: int,
    ny: int,
    allowed_time_indices: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    spatial_size = nx * ny
    allowed_per_sample = len(allowed_time_indices) * spatial_size
    local = _sample_unique_integers(sample_count * allowed_per_sample, size, rng)
    sample_indices = local // allowed_per_sample
    remainder = local % allowed_per_sample
    local_time = remainder // spatial_size
    cells = remainder % spatial_size
    time_indices = allowed_time_indices[local_time]
    return (
        sample_indices * (times_per_sample * spatial_size)
        + time_indices * spatial_size
        + cells
    ).astype(np.int64)


def _corrected_test4_bundle(
    dataset_root: Path,
    *,
    time_fraction: float,
    split_seed: int,
    interpolation_size: int,
    extrapolation_size: int,
    pool_size: int,
) -> UnifiedBundle:
    """Build a bounded-memory seven-input bundle from corrected Test 4 files."""

    dataset_dir = dataset_root / "test4"
    sample_paths = sorted(dataset_dir.glob("sample_*.npz"))
    if not sample_paths:
        raise FileNotFoundError(
            f"Corrected Test 4 samples were not found in {dataset_dir}"
        )
    if pool_size < 2 * max(LABEL_BUDGETS):
        raise ValueError(
            "corrected Test 4 pool size must be at least twice the largest "
            f"label budget ({2 * max(LABEL_BUDGETS)})"
        )

    required = {
        "times",
        "x",
        "y",
        "pressure",
        "water_saturation",
        "oil_saturation",
        "permeability",
        "porosity",
        "metadata_json",
    }
    with np.load(sample_paths[0], allow_pickle=False) as first:
        missing = required.difference(first.files)
        if missing:
            raise ValueError(
                f"Corrected Test 4 sample {sample_paths[0]} is missing: "
                f"{sorted(missing)}"
            )
        times = np.asarray(first["times"], dtype=np.float64)
        x = np.asarray(first["x"], dtype=np.float64)
        y = np.asarray(first["y"], dtype=np.float64)
        reference_permeability = np.asarray(first["permeability"], dtype=np.float64)
        reference_porosity = np.asarray(first["porosity"], dtype=np.float64)
        first_metadata = json.loads(str(first["metadata_json"].item()))

    if len(times) < 2 or not np.isclose(times[0], 0.0):
        raise ValueError("Corrected Test 4 must contain t=0 and later states")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("Corrected Test 4 times must be strictly increasing")
    if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
        raise ValueError("Corrected Test 4 cell centres must be increasing")
    if reference_permeability.shape != (len(x), len(y)):
        raise ValueError("Corrected Test 4 permeability shape is inconsistent")
    if reference_porosity.shape != reference_permeability.shape:
        raise ValueError("Corrected Test 4 porosity shape is inconsistent")
    if not np.allclose(reference_porosity, reference_porosity.flat[0], atol=1e-12):
        raise ValueError("The current PINN protocol requires constant porosity")

    fluids = first_metadata.get("fluids", {})
    water_viscosity = float(fluids.get("water_viscosity", 1.0))
    oil_viscosity = float(fluids.get("oil_viscosity", 3.0))
    full_time = float(times[-1])
    cutoff = time_fraction * full_time
    early_time_indices = np.flatnonzero(times <= cutoff + 1e-12)
    late_time_indices = np.flatnonzero(times > cutoff + 1e-12)
    if not len(early_time_indices) or not len(late_time_indices):
        raise ValueError("Corrected Test 4 temporal split is empty")

    early_pool_size = pool_size // 2
    late_pool_size = pool_size - early_pool_size
    rng = np.random.default_rng(split_seed)
    early_codes = _test4_codes_for_times(
        sample_count=len(sample_paths),
        times_per_sample=len(times),
        nx=len(x),
        ny=len(y),
        allowed_time_indices=early_time_indices,
        size=interpolation_size + early_pool_size,
        rng=rng,
    )
    late_codes = _test4_codes_for_times(
        sample_count=len(sample_paths),
        times_per_sample=len(times),
        nx=len(x),
        ny=len(y),
        allowed_time_indices=late_time_indices,
        size=extrapolation_size + late_pool_size,
        rng=rng,
    )
    codes = np.concatenate((early_codes, late_codes))
    sample_span = len(times) * len(x) * len(y)
    spatial_size = len(x) * len(y)
    sample_indices = codes // sample_span
    remainder = codes % sample_span
    time_indices = remainder // spatial_size
    cells = remainder % spatial_size
    x_indices = cells // len(y)
    y_indices = cells % len(y)

    features = np.empty((len(codes), 7), dtype=np.float32)
    targets = np.empty((len(codes), 3), dtype=np.float32)
    permeability = np.empty(len(codes), dtype=np.float32)
    features[:, 0] = times[time_indices]
    features[:, 1] = x[x_indices]
    features[:, 2] = y[y_indices]

    lower = np.asarray((1.5, 3.0, 0.9, 0.2), dtype=np.float64)
    upper = np.asarray((2.5, 5.0, 1.1, 0.4), dtype=np.float64)
    expected_shape = (len(times), len(x), len(y))
    for sample_index, path in enumerate(sample_paths):
        positions = np.flatnonzero(sample_indices == sample_index)
        if not len(positions):
            continue
        with np.load(path, allow_pickle=False) as data:
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Corrected Test 4 sample {path} is missing {missing}")
            if not (
                np.array_equal(data["times"], times)
                and np.array_equal(data["x"], x)
                and np.array_equal(data["y"], y)
            ):
                raise ValueError(f"Corrected Test 4 grids differ in {path}")
            pressure = np.asarray(data["pressure"], dtype=np.float64)
            swat = np.asarray(data["water_saturation"], dtype=np.float64)
            soil = np.asarray(data["oil_saturation"], dtype=np.float64)
            if pressure.shape != expected_shape or swat.shape != expected_shape:
                raise ValueError(f"Corrected Test 4 field shape differs in {path}")
            if soil.shape != expected_shape:
                raise ValueError(f"Corrected Test 4 oil saturation shape differs in {path}")
            sample_permeability = np.asarray(data["permeability"], dtype=np.float64)
            sample_porosity = np.asarray(data["porosity"], dtype=np.float64)
            if not np.allclose(sample_permeability, reference_permeability):
                raise ValueError(f"Corrected Test 4 permeability differs in {path}")
            if not np.allclose(sample_porosity, reference_porosity):
                raise ValueError(f"Corrected Test 4 porosity differs in {path}")
            metadata = json.loads(str(data["metadata_json"].item()))

        relative = metadata.get("relative_permeability", {})
        parameters = np.asarray(
            (
                relative["water_exponent"],
                relative["oil_exponent"],
                relative["water_endpoint"],
                relative["oil_endpoint"],
            ),
            dtype=np.float64,
        )
        if np.any(parameters < lower - 1e-12) or np.any(parameters > upper + 1e-12):
            raise ValueError(f"Corrected Test 4 parameters lie outside bounds in {path}")
        sample_fluids = metadata.get("fluids", {})
        if not np.isclose(sample_fluids.get("water_viscosity", 1.0), water_viscosity):
            raise ValueError(f"Water viscosity differs in {path}")
        if not np.isclose(sample_fluids.get("oil_viscosity", 3.0), oil_viscosity):
            raise ValueError(f"Oil viscosity differs in {path}")
        boundary = metadata.get("boundary_conditions", {})
        if not np.isclose(boundary.get("left_pressure", 1.0), 1.0):
            raise ValueError(f"Left pressure differs in {path}")
        if not np.isclose(boundary.get("right_pressure", 0.0), 0.0):
            raise ValueError(f"Right pressure differs in {path}")

        ti = time_indices[positions]
        xi = x_indices[positions]
        yi = y_indices[positions]
        features[positions, 3:] = parameters.astype(np.float32)
        targets[positions, 0] = pressure[ti, xi, yi]
        targets[positions, 1] = soil[ti, xi, yi]
        targets[positions, 2] = swat[ti, xi, yi]
        permeability[positions] = reference_permeability[xi, yi]

    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("Corrected Test 4 materialised pool contains non-finite values")
    if not np.allclose(targets[:, 1] + targets[:, 2], 1.0, atol=1e-7):
        raise ValueError("Corrected Test 4 saturations do not satisfy closure")

    interpolation_idx = np.arange(interpolation_size, dtype=np.int64)
    early_pool_start = interpolation_size
    early_pool_end = early_pool_start + early_pool_size
    train_candidates = np.arange(early_pool_start, early_pool_end, dtype=np.int64)
    extrapolation_start = early_pool_end
    extrapolation_idx = np.arange(
        extrapolation_start,
        extrapolation_start + extrapolation_size,
        dtype=np.int64,
    )
    late_pool_start = extrapolation_start + extrapolation_size
    late_pool = np.arange(late_pool_start, len(codes), dtype=np.int64)
    residual_full = np.concatenate((train_candidates, late_pool))

    return UnifiedBundle(
        test_name="test4",
        module=None,
        data_source="corrected",
        dataset_path=str(dataset_dir.resolve()),
        time_cutoff=cutoff,
        full_time=full_time,
        input_lower=(0.0, 0.0, 0.0, 1.5, 3.0, 0.9, 0.2),
        input_upper=(full_time, 1.0, 1.0, 2.5, 5.0, 1.1, 0.4),
        porosity_value=float(reference_porosity.flat[0]),
        water_viscosity=water_viscosity,
        oil_viscosity=oil_viscosity,
        features=features,
        targets=targets,
        permeability=permeability,
        train_candidates=train_candidates,
        residual_candidates_full=residual_full,
        residual_candidates_label=train_candidates,
        interpolation_x=features[interpolation_idx],
        interpolation_y=targets[interpolation_idx],
        extrapolation_x=features[extrapolation_idx],
        extrapolation_y=targets[extrapolation_idx],
    )


def build_unified_bundle(
    test_name: str,
    time_fraction: float = 0.7,
    split_seed: int = 20260801,
    interpolation_size: int = 1500,
    extrapolation_size: int = 1500,
    data_source: str = "legacy",
    corrected_data_root: str | Path | None = None,
    corrected_test4_pool_size: int = 100_000,
) -> UnifiedBundle:
    if not 0.0 < time_fraction < 1.0:
        raise ValueError("time_fraction must be in (0, 1)")
    if test_name not in TEST_NAMES:
        raise ValueError(test_name)
    if data_source not in {"legacy", "corrected"}:
        raise ValueError("data_source must be 'legacy' or 'corrected'")
    if data_source == "corrected" and test_name not in {"test2", "test3", "test4"}:
        raise ValueError(
            "Corrected unified loading is implemented for Tests 2--4"
        )
    if data_source == "corrected" and test_name == "test4":
        if corrected_data_root is None:
            raise ValueError("corrected_data_root is required for corrected data")
        return _corrected_test4_bundle(
            Path(corrected_data_root).expanduser().resolve(),
            time_fraction=time_fraction,
            split_seed=split_seed,
            interpolation_size=interpolation_size,
            extrapolation_size=extrapolation_size,
            pool_size=corrected_test4_pool_size,
        )
    module = None
    dataset_path: Path | None = None
    if test_name == "test1":
        # Test 1 is analytical and has no external dataset dependency.
        module = _HeatEquationModule()
        full_time = DOMAIN_BOUNDS[test_name][1][0]
        input_lower = tuple(float(value) for value in DOMAIN_BOUNDS[test_name][0])
        input_upper = tuple(float(value) for value in DOMAIN_BOUNDS[test_name][1])
        porosity_value = 0.1
        water_viscosity = 1.0
        oil_viscosity = 3.0
    elif data_source == "corrected":
        if corrected_data_root is None:
            raise ValueError("corrected_data_root is required for corrected data")
        dataset_path = (
            Path(corrected_data_root).expanduser().resolve()
            / test_name
            / "solution.npz"
        )
        (
            features,
            targets,
            permeability,
            porosity_value,
            water_viscosity,
            oil_viscosity,
        ) = _corrected_fixed_flow_arrays(test_name, dataset_path)
        full_time = float(np.max(features[:, 0]))
        input_lower = (0.0, 0.0, 0.0)
        input_upper = (full_time, 1.0, 1.0)
    else:
        from paper_jcp.evaluate_fixed_holdout import load_training_module
        from paper_jcp.run_robustness_experiments import (
            build_bundle as build_legacy_bundle,
        )

        module = load_training_module(test_name)
        full_time = DOMAIN_BOUNDS[test_name][1][0]
        input_lower = tuple(float(value) for value in DOMAIN_BOUNDS[test_name][0])
        input_upper = tuple(float(value) for value in DOMAIN_BOUNDS[test_name][1])
        porosity_value = 0.1
        water_viscosity = 1.0
        oil_viscosity = 3.0
    cutoff = time_fraction * full_time

    if test_name == "test1":
        t_grid = np.linspace(0.0, full_time, 161, dtype=np.float32)
        x_grid = np.linspace(0.0, 1.0, 257, dtype=np.float32)
        interp_t = t_grid[t_grid <= cutoff + 1e-8]
        extra_t = t_grid[t_grid > cutoff + 1e-8]
        interpolation_x = np.array(
            np.meshgrid(interp_t, x_grid, indexing="ij")
        ).reshape(2, -1).T.astype(np.float32)
        extrapolation_x = np.array(
            np.meshgrid(extra_t, x_grid, indexing="ij")
        ).reshape(2, -1).T.astype(np.float32)
        with torch.no_grad():
            interpolation_y = module.thermal_conductivity_equation(
                torch.tensor(interpolation_x[:, 0]),
                torch.tensor(interpolation_x[:, 1]),
            ).numpy().astype(np.float32)
            extrapolation_y = module.thermal_conductivity_equation(
                torch.tensor(extrapolation_x[:, 0]),
                torch.tensor(extrapolation_x[:, 1]),
            ).numpy().astype(np.float32)
        return UnifiedBundle(
            test_name=test_name,
            module=module,
            data_source=data_source,
            dataset_path=None,
            time_cutoff=cutoff,
            full_time=full_time,
            input_lower=input_lower,
            input_upper=input_upper,
            porosity_value=porosity_value,
            water_viscosity=water_viscosity,
            oil_viscosity=oil_viscosity,
            features=None,
            targets=None,
            permeability=None,
            train_candidates=None,
            residual_candidates_full=None,
            residual_candidates_label=None,
            interpolation_x=interpolation_x,
            interpolation_y=interpolation_y,
            extrapolation_x=extrapolation_x,
            extrapolation_y=extrapolation_y,
        )

    if data_source == "legacy":
        legacy = build_legacy_bundle(test_name, module)
        features = legacy["features"]
        targets = legacy["targets"]
        permeability = legacy["perm"]

    early = np.flatnonzero(features[:, 0] <= cutoff + 1e-8)
    late = np.flatnonzero(features[:, 0] > cutoff + 1e-8)
    rng = np.random.default_rng(split_seed)
    interpolation_idx = _fixed_split(early, interpolation_size, rng)
    extrapolation_idx = _fixed_split(late, extrapolation_size, rng)
    test_union = np.union1d(interpolation_idx, extrapolation_idx)
    train_candidates = np.setdiff1d(early, interpolation_idx, assume_unique=False)
    residual_full = np.setdiff1d(
        np.arange(len(features)), test_union, assume_unique=False
    )
    residual_label = np.setdiff1d(early, interpolation_idx, assume_unique=False)
    if len(train_candidates) < max(LABEL_BUDGETS):
        raise ValueError(
            f"{test_name} has only {len(train_candidates)} early-time candidates; "
            f"need at least {max(LABEL_BUDGETS)}"
        )
    return UnifiedBundle(
        test_name=test_name,
        module=module,
        data_source=data_source,
        dataset_path=None if dataset_path is None else str(dataset_path),
        time_cutoff=cutoff,
        full_time=full_time,
        input_lower=input_lower,
        input_upper=input_upper,
        porosity_value=porosity_value,
        water_viscosity=water_viscosity,
        oil_viscosity=oil_viscosity,
        features=features,
        targets=targets,
        permeability=permeability,
        train_candidates=train_candidates,
        residual_candidates_full=residual_full,
        residual_candidates_label=residual_label,
        interpolation_x=features[interpolation_idx],
        interpolation_y=targets[interpolation_idx],
        extrapolation_x=features[extrapolation_idx],
        extrapolation_y=targets[extrapolation_idx],
    )


def _interleave(groups: list[np.ndarray], limit: int) -> np.ndarray:
    result: list[int] = []
    positions = [0] * len(groups)
    while len(result) < limit:
        progressed = False
        for group_index, group in enumerate(groups):
            position = positions[group_index]
            if position < len(group):
                result.append(int(group[position]))
                positions[group_index] += 1
                progressed = True
                if len(result) == limit:
                    break
        if not progressed:
            break
    if len(result) < limit:
        raise ValueError(f"Could only construct {len(result)} of {limit} nested points")
    return np.asarray(result, dtype=np.int64)


def nested_label_indices(
    bundle: UnifiedBundle,
    seed: int,
    max_budget: int = max(LABEL_BUDGETS),
    time_bins: int = 10,
) -> np.ndarray:
    if bundle.test_name == "test1":
        raise ValueError("Test 1 labels are generated continuously")
    if time_bins < 1:
        raise ValueError("time_bins must be positive")
    candidates = bundle.train_candidates
    assert candidates is not None and bundle.features is not None
    rng = np.random.default_rng(seed)
    if time_bins == 1:
        return rng.choice(candidates, size=max_budget, replace=False)

    time = bundle.features[candidates, 0]
    edges = np.linspace(0.0, bundle.time_cutoff, time_bins + 1)
    bin_ids = np.clip(np.digitize(time, edges[1:-1]), 0, time_bins - 1)
    per_bin = math.ceil(max_budget / time_bins) + 2
    groups = []
    for bin_index in range(time_bins):
        local = candidates[bin_ids == bin_index]
        if len(local) == 0:
            continue
        count = min(per_bin, len(local))
        groups.append(rng.choice(local, size=count, replace=False))
    return _interleave(groups, max_budget)


def nested_heat_labels(
    bundle: UnifiedBundle,
    seed: int,
    max_budget: int = max(LABEL_BUDGETS),
    time_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    per_bin = math.ceil(max_budget / time_bins) + 2
    edges = np.linspace(0.0, bundle.time_cutoff, time_bins + 1)
    for index in range(time_bins):
        t = rng.uniform(edges[index], edges[index + 1], per_bin)
        x = rng.uniform(0.0, 1.0, per_bin)
        groups.append(np.column_stack((t, x)).astype(np.float32))
    rows = []
    positions = [0] * len(groups)
    while len(rows) < max_budget:
        for index, group in enumerate(groups):
            if positions[index] < len(group):
                rows.append(group[positions[index]])
                positions[index] += 1
                if len(rows) == max_budget:
                    break
    x_data = np.asarray(rows, dtype=np.float32)
    with torch.no_grad():
        y_data = bundle.module.thermal_conductivity_equation(
            torch.tensor(x_data[:, 0]), torch.tensor(x_data[:, 1])
        ).numpy().astype(np.float32)
    return x_data, y_data


class ResidualSampler:
    def __init__(
        self,
        bundle: UnifiedBundle,
        seed: int,
        domain: str = "full",
        exclude_indices: np.ndarray | None = None,
    ) -> None:
        if domain not in {"full", "label"}:
            raise ValueError("residual domain must be 'full' or 'label'")
        self.bundle = bundle
        self.rng = np.random.default_rng(seed)
        self.domain = domain
        self.exclude_indices = exclude_indices
        self.candidates: np.ndarray | None = None
        if bundle.test_name != "test1":
            candidates = (
                bundle.residual_candidates_full
                if domain == "full"
                else bundle.residual_candidates_label
            )
            assert candidates is not None
            if exclude_indices is not None and len(exclude_indices):
                candidates = np.setdiff1d(
                    candidates, exclude_indices, assume_unique=False
                )
            self.candidates = candidates

    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray | None]:
        if size <= 0:
            raise ValueError("Residual batch size must be positive")
        if self.bundle.test_name == "test1":
            end = self.bundle.full_time if self.domain == "full" else self.bundle.time_cutoff
            points = np.column_stack(
                (
                    self.rng.uniform(0.0, end, size),
                    self.rng.uniform(0.0, 1.0, size),
                )
            ).astype(np.float32)
            return points, None

        candidates = self.candidates
        assert candidates is not None
        replace = size > len(candidates)
        indices = self.rng.choice(candidates, size=size, replace=replace)
        assert self.bundle.features is not None and self.bundle.permeability is not None
        return self.bundle.features[indices], self.bundle.permeability[indices]


def heat_loss_components(
    model: nn.Module,
    module: object,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_collocation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = model(x_data).reshape(-1)
    loss_data = torch.mean((prediction - y_data.reshape(-1)) ** 2)

    t = x_collocation[:, 0:1].clone().detach().requires_grad_(True)
    x = x_collocation[:, 1:2].clone().detach().requires_grad_(True)
    interior = torch.cat((t, x), dim=1)
    initial = torch.cat((torch.zeros_like(t), x), dim=1)
    left = torch.cat((t, torch.zeros_like(x)), dim=1)
    right = torch.cat((t, torch.ones_like(x)), dim=1)
    initial_target = module.thermal_conductivity_equation(
        torch.zeros_like(t.reshape(-1)), x.reshape(-1)
    )
    loss_ibc = torch.mean(
        (model(initial).reshape(-1) - initial_target) ** 2
        + (model(left).reshape(-1) - model(right).reshape(-1)) ** 2
    )

    temperature = model(interior).reshape(-1)
    ones = torch.ones_like(temperature)
    dt = torch.autograd.grad(temperature, t, ones, create_graph=True)[0].reshape(-1)
    dx = torch.autograd.grad(temperature, x, ones, create_graph=True)[0]
    dxx = torch.autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0].reshape(-1)
    loss_pde = torch.mean((dt - dxx) ** 2)
    return loss_data, loss_ibc, loss_pde


def flow_residuals(
    test_name: str,
    model: nn.Module,
    points: torch.Tensor,
    permeability: torch.Tensor,
    *,
    porosity: float = 0.1,
    water_viscosity: float = 1.0,
    oil_viscosity: float = 3.0,
) -> tuple[torch.Tensor, ...]:
    t = points[:, 0:1].clone().detach().requires_grad_(True)
    x = points[:, 1:2].clone().detach().requires_grad_(True)
    y = points[:, 2:3].clone().detach().requires_grad_(True)
    if test_name == "test4":
        physical_parameters = [points[:, index : index + 1] for index in range(3, 7)]
        model_input = torch.cat((t, x, y, *physical_parameters), dim=1)
        pwat, poil, kwat, koil = physical_parameters
    else:
        model_input = torch.cat((t, x, y), dim=1)
        pwat = torch.full_like(t, 2.0)
        poil = torch.full_like(t, 4.0)
        kwat = torch.full_like(t, 1.0)
        koil = torch.full_like(t, 0.3)

    result = model(model_input)
    ones = torch.ones_like(result[:, 0])

    def grad(output: torch.Tensor, variable: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            output,
            variable,
            ones,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0].reshape(-1)

    pressure_x = grad(result[:, 0], x)
    pressure_y = grad(result[:, 0], y)
    swat = result[:, 2]
    soil = result[:, 1]
    perm = permeability.reshape(-1)
    water_mass = porosity * grad(swat, t) + grad(result[:, 5], x) + grad(result[:, 6], y)
    oil_mass = porosity * grad(soil, t) + grad(result[:, 3], x) + grad(result[:, 4], y)
    water_factor = (
        kwat.reshape(-1) * perm * swat ** pwat.reshape(-1) / water_viscosity
    )
    oil_factor = (
        koil.reshape(-1) * perm * soil ** poil.reshape(-1) / oil_viscosity
    )
    darcy_wx = result[:, 5] + water_factor * pressure_x
    darcy_wy = result[:, 6] + water_factor * pressure_y
    darcy_ox = result[:, 3] + oil_factor * pressure_x
    darcy_oy = result[:, 4] + oil_factor * pressure_y
    closure = swat + soil - 1.0
    return water_mass, oil_mass, darcy_wx, darcy_wy, darcy_ox, darcy_oy, closure


def flow_loss_components(
    bundle: UnifiedBundle,
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_collocation: torch.Tensor,
    permeability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    test_name = bundle.test_name
    prediction = model(x_data)
    loss_data = torch.mean(
        (prediction[:, 0] - y_data[:, 0]) ** 2
        + (prediction[:, 1] - y_data[:, 1]) ** 2
        + (prediction[:, 2] - y_data[:, 2]) ** 2
    )

    t = x_collocation[:, 0]
    x = x_collocation[:, 1]
    y = x_collocation[:, 2]
    if test_name == "test4":
        parameters = [x_collocation[:, index] for index in range(3, 7)]
        left = torch.stack((t, torch.zeros_like(x), y, *parameters), dim=1)
        right = torch.stack((t, torch.ones_like(x), y, *parameters), dim=1)
        initial = torch.stack((torch.zeros_like(t), x, y, *parameters), dim=1)
    else:
        left = torch.stack((t, torch.zeros_like(x), y), dim=1)
        right = torch.stack((t, torch.ones_like(x), y), dim=1)
        initial = torch.stack((torch.zeros_like(t), x, y), dim=1)
    loss_ibc_terms = (
        (model(left)[:, 0] - 1.0) ** 2
        + model(right)[:, 0] ** 2
        + (model(initial)[:, 1] - 1.0) ** 2
        + model(initial)[:, 2] ** 2
    )
    if bundle.data_source == "corrected":
        if test_name == "test4":
            bottom = torch.stack((t, x, torch.zeros_like(y), *parameters), dim=1)
            top = torch.stack((t, x, torch.ones_like(y), *parameters), dim=1)
        else:
            bottom = torch.stack((t, x, torch.zeros_like(y)), dim=1)
            top = torch.stack((t, x, torch.ones_like(y)), dim=1)
        left_result = model(left)
        bottom_result = model(bottom)
        top_result = model(top)
        # Corrected FV boundary conditions: pure-water inflow at x=0
        # (zero incoming oil flux) and impermeable y boundaries for both phases.
        loss_ibc_terms = (
            loss_ibc_terms
            + left_result[:, 3] ** 2
            + bottom_result[:, 4] ** 2
            + bottom_result[:, 6] ** 2
            + top_result[:, 4] ** 2
            + top_result[:, 6] ** 2
        )
    loss_ibc = torch.mean(loss_ibc_terms)

    residuals = flow_residuals(
        test_name,
        model,
        x_collocation,
        permeability,
        porosity=bundle.porosity_value,
        water_viscosity=bundle.water_viscosity,
        oil_viscosity=bundle.oil_viscosity,
    )
    loss_pde = torch.mean(sum(residual**2 for residual in residuals))
    return loss_data, loss_ibc, loss_pde


def loss_components(
    bundle: UnifiedBundle,
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_collocation: torch.Tensor,
    permeability: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bundle.test_name == "test1":
        return heat_loss_components(
            model, bundle.module, x_data, y_data, x_collocation
        )
    if permeability is None:
        raise ValueError("Flow losses require permeability values")
    return flow_loss_components(
        bundle, model, x_data, y_data, x_collocation, permeability
    )


class VanillaPINNLossBalancer:
    """Unweighted sum of the data, IBC, and PDE loss components."""

    @staticmethod
    def compute_weights(
        model: nn.Module,
        loss_mse: torch.Tensor,
        loss_ibc: torch.Tensor,
        loss_pde: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        del model, loss_ibc, loss_pde
        weight = loss_mse.detach().new_tensor(1.0)
        diagnostics = {
            "a_mse": 1.0,
            "a_ibc": 1.0,
            "a_pde": 1.0,
            "w_mse": 1.0,
            "w_ibc": 1.0,
            "w_pde": 1.0,
        }
        return weight, weight.clone(), weight.clone(), diagnostics


def make_balancer(
    mode: str,
    config: TrainConfig,
) -> VanillaPINNLossBalancer | HierarchicalOrthogonalPINNLossBalancer:
    if mode == "vanilla":
        return VanillaPINNLossBalancer()
    if mode not in {"h", "ho"}:
        raise ValueError("balancer must be 'vanilla', 'h', or 'ho'")
    return HierarchicalOrthogonalPINNLossBalancer(
        L0_mse=config.l0_data,
        L0_ibc=config.l0_ibc,
        use_orthogonal_factor=mode == "ho",
        orthogonal_floor=config.orthogonal_floor,
    )


def _optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_learning_rate
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def train_pinn(
    bundle: UnifiedBundle,
    x_data_np: np.ndarray,
    y_data_np: np.ndarray,
    label_indices: np.ndarray | None,
    label_budget: int,
    seed: int,
    balance_mode: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_path: Path,
    history_path: Path,
) -> tuple[UnifiedPINN, float]:
    device = resolve_device(train_config.device)
    configure_torch_runtime(seed + 1000, device, train_config.deterministic)
    model = build_model(bundle, model_config).to(device)
    optimizer = _optimizer(model, train_config)
    scheduler = _scheduler(optimizer, train_config)
    balancer = make_balancer(balance_mode, train_config)
    residual_size = max(1, int(math.ceil(label_budget * train_config.collocation_ratio)))
    sampler = ResidualSampler(
        bundle,
        seed + 500,
        train_config.residual_domain,
        exclude_indices=label_indices,
    )
    validation_sampler = ResidualSampler(
        bundle,
        seed + 50_000,
        train_config.residual_domain,
        exclude_indices=label_indices,
    )
    validation_np, validation_perm_np = validation_sampler.sample(
        train_config.physics_validation_size
    )

    x_data = torch.as_tensor(x_data_np, dtype=torch.float32, device=device)
    y_data = torch.as_tensor(y_data_np, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(
        validation_np, dtype=torch.float32, device=device
    )
    validation_perm = (
        None
        if validation_perm_np is None
        else torch.as_tensor(validation_perm_np, dtype=torch.float32, device=device)
    )
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in trange(train_config.epochs):
        residual_np, permeability_np = sampler.sample(residual_size)
        x_collocation = torch.as_tensor(
            residual_np, dtype=torch.float32, device=device
        )
        permeability = (
            None
            if permeability_np is None
            else torch.as_tensor(
                permeability_np, dtype=torch.float32, device=device
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss_data, loss_ibc, loss_pde = loss_components(
            bundle,
            model,
            x_data,
            y_data,
            x_collocation,
            permeability,
        )
        w_data, w_ibc, w_pde, diagnostics = balancer.compute_weights(
            model, loss_data, loss_ibc, loss_pde
        )
        balanced = w_data * loss_data + w_ibc * loss_ibc + w_pde * loss_pde
        raw = loss_data + loss_ibc + loss_pde
        balanced.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        criterion = balanced if train_config.checkpoint_criterion == "balanced" else raw
        value = float(criterion.detach())
        if value < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())

        should_log = epoch == 0 or (epoch + 1) % train_config.log_every == 0
        if should_log or epoch + 1 == train_config.epochs:
            val_losses = loss_components(
                bundle,
                model,
                x_data,
                y_data,
                validation_x,
                validation_perm,
            )
            print({
                    "epoch": epoch + 1,
                    "loss_data": float(loss_data.detach()),
                    "loss_ibc": float(loss_ibc.detach()),
                    "loss_pde": float(loss_pde.detach()),
                    "raw_total": float(raw.detach()),
                    "weight_data": float(w_data),
                    "weight_ibc": float(w_ibc),
                    "weight_pde": float(w_pde),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                })
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss_data": float(loss_data.detach()),
                    "loss_ibc": float(loss_ibc.detach()),
                    "loss_pde": float(loss_pde.detach()),
                    "raw_total": float(raw.detach()),
                    "balanced_total": float(balanced.detach()),
                    "weight_data": float(w_data),
                    "weight_ibc": float(w_ibc),
                    "weight_pde": float(w_pde),
                    "weighted_data": float((w_data * loss_data).detach()),
                    "weighted_ibc": float((w_ibc * loss_ibc).detach()),
                    "weighted_pde": float((w_pde * loss_pde).detach()),
                    "physics_val_ibc": float(val_losses[1].detach()),
                    "physics_val_pde": float(val_losses[2].detach()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **diagnostics,
                }
            )

    if best_state is None:
        raise RuntimeError("No PINN checkpoint was produced")
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    portable_state = {
        name: value.detach().cpu().clone() for name, value in best_state.items()
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": portable_state,
            "test_name": bundle.test_name,
            "data_source": bundle.data_source,
            "dataset_path": bundle.dataset_path,
            "input_lower": bundle.input_lower,
            "input_upper": bundle.input_upper,
            "porosity": bundle.porosity_value,
            "water_viscosity": bundle.water_viscosity,
            "oil_viscosity": bundle.oil_viscosity,
            "balance_mode": balance_mode,
            "label_budget": label_budget,
            "seed": seed,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "resolved_device": str(device),
            "best_value": best_value,
        },
        checkpoint_path,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    metadata_path = history_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "elapsed_seconds": elapsed,
                "data_source": bundle.data_source,
                "dataset_path": bundle.dataset_path,
                "input_lower": bundle.input_lower,
                "input_upper": bundle.input_upper,
                "porosity": bundle.porosity_value,
                "water_viscosity": bundle.water_viscosity,
                "oil_viscosity": bundle.oil_viscosity,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "resolved_device": str(device),
            },
            indent=2,
        )
    )
    return model, elapsed


def predict_pinn(test_name: str, model: nn.Module, features: np.ndarray) -> np.ndarray:
    parts = []
    device = next(model.parameters()).device
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(tensor), 4096):
            parts.append(model(tensor[start : start + 4096]).cpu().numpy())
    prediction = np.concatenate(parts, axis=0)
    return prediction.reshape(-1) if test_name == "test1" else prediction[:, :3]


def evaluation_metrics(test_name: str, truth: np.ndarray, prediction: np.ndarray) -> dict:
    if test_name == "test1":
        result = compute_scalar_metrics(truth.reshape(-1), prediction.reshape(-1))
        denominator = np.linalg.norm(truth.reshape(-1)) + 1e-12
        result["relative_L2"] = float(
            np.linalg.norm(truth.reshape(-1) - prediction.reshape(-1)) / denominator
        )
        target_std = float(np.std(truth))
        result["NRMSE_std"] = result["RMSE"] / (target_std + 1e-12)
        return result

    result = compute_multioutput_metrics(truth, prediction, TARGET_NAMES)
    relative_values = []
    nrmse_values = []
    for index, name in enumerate(TARGET_NAMES):
        denominator = np.linalg.norm(truth[:, index]) + 1e-12
        relative = float(
            np.linalg.norm(truth[:, index] - prediction[:, index]) / denominator
        )
        nrmse = result[f"{name}_RMSE"] / (float(np.std(truth[:, index])) + 1e-12)
        result[f"{name}_relative_L2"] = relative
        result[f"{name}_NRMSE_std"] = nrmse
        relative_values.append(relative)
        nrmse_values.append(nrmse)
    result["mean_relative_L2"] = float(np.mean(relative_values))
    result["mean_NRMSE_std"] = float(np.mean(nrmse_values))
    result["saturation_bounds_violation"] = float(
        np.mean(
            (prediction[:, 1] < -1e-3)
            | (prediction[:, 1] > 1.0 + 1e-3)
            | (prediction[:, 2] < -1e-3)
            | (prediction[:, 2] > 1.0 + 1e-3)
        )
    )
    result["closure_MAE"] = float(np.mean(np.abs(prediction[:, 1] + prediction[:, 2] - 1.0)))
    return result


def configuration_id(
    balance_mode: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> str:
    suffix = ""
    if train_config.epochs != 10_000:
        suffix += f"_e{train_config.epochs}"
    if train_config.device != "cpu":
        suffix += f"_{train_config.device.replace(':', '-')}"
    if not train_config.deterministic:
        suffix += "_nondeterministic"
    return (
        f"{balance_mode}_w{model_config.hidden_dim}_d{model_config.depth}_"
        f"{model_config.activation}_lr{train_config.learning_rate:g}_"
        f"l0d{train_config.l0_data:g}_l0b{train_config.l0_ibc:g}_"
        f"{train_config.optimizer}_{train_config.scheduler}_"
        f"nr{train_config.collocation_ratio:g}_{train_config.residual_domain}{suffix}"
    )
