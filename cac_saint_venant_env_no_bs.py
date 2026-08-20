#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saint-Venant PDE environment for pure SAC without backstepping.

This environment does NOT compute backstepping control.
This environment does NOT solve the backstepping kernel.
The SAC action directly represents the downstream physical gate opening U_L.

Main features:
    1) No clipping of water depth H in PDE computation.
    2) Internal PDE state is stored in float64.
    3) Observation is normalized by default:
           obs = [(H-Hbar)/H_scale, (Q-Qbar)/Q_scale]
    4) SAC action a in [-1,1] is mapped to U_L in [U_min, U_max].
    5) Characteristic ghost-cell boundary conditions.
    6) CFL check before every PDE sub-step.
    7) Nonphysical states are detected immediately after each PDE sub-step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# ============================================================
# Parameters
# ============================================================

@dataclass
class CACParams:
    L: float
    B: float
    g: float
    I: float
    K: float
    mu0: float
    muL: float

    z_up: float
    U0_bar: float
    U_L_bar: float
    h_s: float

    Q_bar: float
    H_bar_0: float
    H_bar_L: float

    H_min: float = 0.20
    max_state_value: float = 1e10


# ============================================================
# Saint-Venant model functions
# ============================================================

def area(H: np.ndarray, p: CACParams) -> np.ndarray:
    return p.B * H


def hydraulic_radius(H: np.ndarray, p: CACParams) -> np.ndarray:
    return p.B * H / (p.B + 2.0 * H)


def friction_slope(H: np.ndarray, Q: np.ndarray, p: CACParams) -> np.ndarray:
    A = area(H, p)
    R = hydraulic_radius(H, p)
    return Q * np.abs(Q) / (p.K**2 * A**2 * R**(4.0 / 3.0))


def flux(H: np.ndarray, Q: np.ndarray, p: CACParams) -> np.ndarray:
    F1 = Q / p.B
    F2 = Q**2 / (p.B * H) + 0.5 * p.g * p.B * H**2
    return np.stack([F1, F2], axis=-1)


def source(H: np.ndarray, Q: np.ndarray, p: CACParams) -> np.ndarray:
    J = friction_slope(H, Q, p)
    return np.stack(
        [
            np.zeros_like(H),
            p.g * p.B * H * (p.I - J),
        ],
        axis=-1,
    )


def characteristic_speeds(
    H: np.ndarray,
    Q: np.ndarray,
    p: CACParams,
) -> Tuple[np.ndarray, np.ndarray]:
    V = Q / (p.B * H)
    c = np.sqrt(p.g * H)
    return V + c, V - c


def max_wave_speed(H: np.ndarray, Q: np.ndarray, p: CACParams) -> float:
    lp, lm = characteristic_speeds(H, Q, p)
    return float(np.max(np.maximum(np.abs(lp), np.abs(lm))))


def rusanov_flux(left: np.ndarray, right: np.ndarray, p: CACParams) -> np.ndarray:
    HL, QL = left[..., 0], left[..., 1]
    HR, QR = right[..., 0], right[..., 1]

    FL = flux(HL, QL, p)
    FR = flux(HR, QR, p)

    lpL, lmL = characteristic_speeds(HL, QL, p)
    lpR, lmR = characteristic_speeds(HR, QR, p)

    a = np.maximum.reduce(
        [
            np.abs(lpL),
            np.abs(lmL),
            np.abs(lpR),
            np.abs(lmR),
        ]
    )

    return 0.5 * (FL + FR) - 0.5 * a[..., None] * (right - left)


def _bisect_root(
    func,
    lo: float,
    hi: float,
    max_iter: int = 60,
    tol: float = 1e-10,
):
    flo = func(lo)
    fhi = func(hi)

    if not np.isfinite(flo) or not np.isfinite(fhi):
        return None

    if flo == 0.0:
        return lo

    if fhi == 0.0:
        return hi

    if flo * fhi > 0.0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = func(mid)

        if not np.isfinite(fmid):
            return None

        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid

        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


def _overflow_safe_downstream_depth(Qb: float, p: CACParams, U_L: float) -> float:
    """
    Compute:

        H = (Qb^2/(2gB^2muL^2))^(1/3) + h_s + U_L

    without explicitly forming Qb**2.
    """
    denom = 2.0 * p.g * p.B**2 * p.muL**2
    return (abs(Qb) ** (2.0 / 3.0)) / (denom ** (1.0 / 3.0)) + p.h_s + U_L


# ============================================================
# Optional helper: compute steady Hbar from steady ODE
# ============================================================

def steady_H_ode_rhs(H: float, Qbar: float, p: CACParams) -> float:
    """
    Steady-state ODE:

        H_x =
            g B H (I - J(H,Qbar))
            /
            (g B H - Qbar^2/(B H^2))

    This function assumes H > 0.
    """
    H_arr = np.array([H], dtype=np.float64)
    Q_arr = np.array([Qbar], dtype=np.float64)

    J = friction_slope(H_arr, Q_arr, p)[0]

    numerator = p.g * p.B * H * (p.I - J)
    denominator = p.g * p.B * H - Qbar**2 / (p.B * H**2)

    return numerator / denominator


def compute_steady_H_profile_rk4(
    X_grid: np.ndarray,
    p: CACParams,
    H0: Optional[float] = None,
    Qbar: Optional[float] = None,
) -> np.ndarray:
    """
    Compute a steady water-depth profile Hbar(x) by RK4 integration.

    Use this in the SAC main program instead of linear interpolation
    if you want the reference profile to satisfy the steady ODE.

    Example:
        Hbar = compute_steady_H_profile_rk4(X, p, H0=p.H_bar_0, Qbar=p.Q_bar)
    """
    X_grid = np.asarray(X_grid, dtype=np.float64)

    if H0 is None:
        H0 = p.H_bar_0

    if Qbar is None:
        Qbar = p.Q_bar

    Hbar = np.zeros_like(X_grid, dtype=np.float64)
    Hbar[0] = float(H0)

    for i in range(len(X_grid) - 1):
        dx = X_grid[i + 1] - X_grid[i]
        H = Hbar[i]

        k1 = steady_H_ode_rhs(H, Qbar, p)
        k2 = steady_H_ode_rhs(H + 0.5 * dx * k1, Qbar, p)
        k3 = steady_H_ode_rhs(H + 0.5 * dx * k2, Qbar, p)
        k4 = steady_H_ode_rhs(H + dx * k3, Qbar, p)

        Hbar[i + 1] = H + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.isfinite(Hbar[i + 1]) or Hbar[i + 1] <= 0.0:
            raise RuntimeError(
                f"Invalid steady Hbar at grid {i+1}: H={Hbar[i+1]}"
            )

    return Hbar


# ============================================================
# Environment
# ============================================================

class CACSaintVenantPDE1D(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        params: CACParams,
        X_grid: np.ndarray,
        Hbar: np.ndarray,
        Qbar: np.ndarray,
        H_init: np.ndarray,
        Q_init: np.ndarray,
        sensing_noise_func: Callable[[np.ndarray], np.ndarray] = lambda obs: obs,
        reset_init_condition_func: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
        sensing_loc: str = "full",
        control_type: str = "Gate",
        sensing_type: Optional[str] = None,
        limit_pde_state_size: bool = False,
        max_state_value: float = 1e10,
        min_control_value: float = 0.0,
        max_control_value: float = 2.0,
        control_sample_rate: float = 4.0,
        reward_class: Optional[Any] = None,
        normalize: bool = True,
        observation_mode: str = "normalized_error",
        H_scale: float = 0.30,
        Q_scale: float = 3.00,
        obs_clip: float = 50.0,
        T: float = 4500.0,
        dt: float = 4.0,
        cfl: float = 0.45,
        check_cfl: bool = True,
        well_balanced: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.params = params

        self.X_grid = np.asarray(X_grid, dtype=np.float64)
        self.x = self.X_grid

        self.Nx = len(self.X_grid) - 1
        self.nx = self.Nx
        self.N = len(self.X_grid)

        self.dx = float(self.X_grid[1] - self.X_grid[0])

        self.Hbar = np.asarray(Hbar, dtype=np.float64)
        self.Qbar = np.asarray(Qbar, dtype=np.float64)
        self.H_init = np.asarray(H_init, dtype=np.float64)
        self.Q_init = np.asarray(Q_init, dtype=np.float64)

        self.T = float(T)
        self.dt = float(dt)
        self.nt = int(np.ceil(self.T / self.dt)) + 2

        self.cfl = float(cfl)
        self.check_cfl = bool(check_cfl)

        self.well_balanced = bool(well_balanced)

        self.sensing_noise_func = sensing_noise_func
        self.reset_init_condition_func = reset_init_condition_func
        self.sensing_loc = sensing_loc
        self.control_type = control_type
        self.sensing_type = sensing_type

        self.limit_pde_state_size = bool(limit_pde_state_size)
        self.max_state_value = float(max_state_value)

        self.min_control_value = float(min_control_value)
        self.max_control_value = float(max_control_value)
        self.control_sample_rate = float(control_sample_rate)

        self.reward_class = reward_class

        self.normalize_action = bool(normalize)

        self.observation_mode = observation_mode
        self.H_scale = float(H_scale)
        self.Q_scale = float(Q_scale)
        self.obs_clip = float(obs_clip)

        self._force_truncate = False

        # ----------------------------------------------------
        # Action space
        # ----------------------------------------------------
        if self.normalize_action:
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
            )

            def map_action(action, _):
                a = float(np.asarray(action).reshape(-1)[0])
                a = np.clip(a, -1.0, 1.0)
                return self.min_control_value + 0.5 * (a + 1.0) * (
                    self.max_control_value - self.min_control_value
                )

            self.normalize = map_action

        else:
            self.action_space = spaces.Box(
                low=np.array([self.min_control_value], dtype=np.float32),
                high=np.array([self.max_control_value], dtype=np.float32),
            )

            def map_action(action, _):
                u = float(np.asarray(action).reshape(-1)[0])
                return float(np.clip(u, self.min_control_value, self.max_control_value))

            self.normalize = map_action

        # ----------------------------------------------------
        # Observation space
        # ----------------------------------------------------
        if self.sensing_loc == "full":
            obs_dim = 2 * (self.nx + 1)
        elif self.sensing_loc in ["collocated", "opposite"]:
            obs_dim = 2
        else:
            raise Exception("Invalid sensing_loc. Use 'full', 'collocated', or 'opposite'.")

        if self.observation_mode == "normalized_error":
            obs_low = -self.obs_clip
            obs_high = self.obs_clip
        elif self.observation_mode == "raw":
            obs_low = -self.max_state_value
            obs_high = self.max_state_value
        else:
            raise Exception("observation_mode must be 'normalized_error' or 'raw'.")

        self.observation_space = spaces.Box(
            low=np.full(obs_dim, obs_low, dtype=np.float32),
            high=np.full(obs_dim, obs_high, dtype=np.float32),
        )

        if self.control_type != "Gate":
            raise Exception("For Saint-Venant, control_type should be 'Gate'.")

        # ----------------------------------------------------
        # Well-balanced residual
        # Usually not used because well_balanced=False.
        # ----------------------------------------------------
        p = self.params
        star_left = np.array([self.Hbar[0], p.Q_bar], dtype=np.float64)
        star_right = np.array([self.Hbar[-1], p.Q_bar], dtype=np.float64)
        star_middle = np.column_stack([self.Hbar, self.Qbar])
        star_ext = np.vstack([star_left, star_middle, star_right])

        self.star_flux = rusanov_flux(star_ext[:-1], star_ext[1:], p)
        self.star_src = source(self.Hbar, self.Qbar, p)

        self.H = np.zeros(self.nx + 1, dtype=np.float64)
        self.Q = np.zeros(self.nx + 1, dtype=np.float64)

        self.u = np.zeros(
            (self.nt, 2 * (self.nx + 1)),
            dtype=np.float64,
        )

        self.time_index = 0
        self.time = 0.0
        self.last_control = self.params.U_L_bar

    # ========================================================
    # Observation helpers
    # ========================================================

    def _state_vector(self) -> np.ndarray:
        return np.concatenate([self.H, self.Q])

    def _raw_observation_vector(self) -> np.ndarray:
        raw = self._state_vector()

        if self.sensing_loc == "full":
            return raw

        if self.sensing_loc == "collocated":
            return raw[[self.nx, 2 * self.nx + 1]]

        if self.sensing_loc == "opposite":
            return raw[[0, self.nx + 1]]

        raise RuntimeError("Invalid sensing_loc.")

    def _normalized_observation_vector(self) -> np.ndarray:
        H_err = (self.H - self.Hbar) / self.H_scale
        Q_err = (self.Q - self.Qbar) / self.Q_scale
        obs_full = np.concatenate([H_err, Q_err])

        if self.sensing_loc == "full":
            return obs_full

        if self.sensing_loc == "collocated":
            return obs_full[[self.nx, 2 * self.nx + 1]]

        if self.sensing_loc == "opposite":
            return obs_full[[0, self.nx + 1]]

        raise RuntimeError("Invalid sensing_loc.")

    def _get_obs(self) -> np.ndarray:
        if self.observation_mode == "normalized_error":
            obs = self._normalized_observation_vector()
            obs = self.sensing_noise_func(obs)
            obs = np.asarray(obs, dtype=np.float64)
            obs = np.nan_to_num(obs, nan=0.0, posinf=self.obs_clip, neginf=-self.obs_clip)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)
            return obs.astype(np.float32)

        obs = self._raw_observation_vector()
        obs = self.sensing_noise_func(obs)
        obs = np.asarray(obs, dtype=np.float64)
        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=self.max_state_value,
            neginf=-self.max_state_value,
        )
        obs = np.clip(obs, -self.max_state_value, self.max_state_value)
        return obs.astype(np.float32)

    # ========================================================
    # State validity
    # ========================================================

    def _bad_state(self) -> bool:
        if self._force_truncate:
            return True

        if not np.all(np.isfinite(self.H)):
            return True

        if not np.all(np.isfinite(self.Q)):
            return True

        if np.any(self.H <= 0.0):
            return True

        if self.limit_pde_state_size:
            state = self._state_vector()
            state_norm = np.linalg.norm(state, 2)

            if not np.isfinite(state_norm):
                return True

            if state_norm >= self.max_state_value:
                return True

        return False

    def _cfl_ok(self, dt: float) -> bool:
        if not self.check_cfl:
            return True

        if self._bad_state():
            return False

        try:
            amax = max_wave_speed(self.H, self.Q, self.params)
        except Exception:
            return False

        if not np.isfinite(amax) or amax <= 0.0:
            return False

        dt_cfl = self.cfl * self.dx / amax

        return dt <= dt_cfl + 1e-12

    # ========================================================
    # Boundary ghost states
    # ========================================================

    def _build_boundary_states(self, control):
        p = self.params

        U_L = self.normalize(control, self.max_control_value)
        self.last_control = U_L

        H0 = float(self.H[0])
        Q0 = float(self.Q[0])
        HL = float(self.H[-1])
        QL = float(self.Q[-1])

        if (
            not np.isfinite(H0)
            or not np.isfinite(Q0)
            or not np.isfinite(HL)
            or not np.isfinite(QL)
            or H0 <= 0.0
            or HL <= 0.0
        ):
            return (
                np.array([H0, Q0], dtype=np.float64),
                np.array([HL, QL], dtype=np.float64),
            )

        # ----------------------------------------------------
        # Upstream boundary x=0
        # Gate equation + outgoing characteristic.
        # ----------------------------------------------------
        V0 = Q0 / (p.B * H0)
        c0 = np.sqrt(p.g * H0)
        R_minus_int = V0 - 2.0 * c0

        def upstream_gate_Q(Hb):
            head = 2.0 * p.g * (p.z_up - Hb)
            if head < 0.0:
                return np.nan
            return p.U0_bar * p.B * p.mu0 * np.sqrt(head)

        def upstream_eq(Hb):
            if Hb <= 0.0:
                return np.nan

            Qb = upstream_gate_Q(Hb)

            if not np.isfinite(Qb):
                return np.nan

            return Qb / (p.B * Hb) - 2.0 * np.sqrt(p.g * Hb) - R_minus_int

        eps = 1e-8
        H_lo = eps
        H_hi = max(p.z_up - eps, 10.0 * eps)

        H_left = _bisect_root(upstream_eq, H_lo, H_hi)

        if H_left is None or not np.isfinite(H_left):
            H_left = H0

        Q_left = upstream_gate_Q(H_left)

        if not np.isfinite(Q_left):
            Q_left = Q0

        left_state = np.array([H_left, Q_left], dtype=np.float64)

        # ----------------------------------------------------
        # Downstream boundary x=L
        # Gate equation + outgoing characteristic.
        # ----------------------------------------------------
        VL = QL / (p.B * HL)
        cL = np.sqrt(p.g * HL)
        R_plus_int = VL + 2.0 * cL

        def downstream_gate_H(Qb):
            return _overflow_safe_downstream_depth(Qb, p, U_L)

        def downstream_eq(Qb):
            Hb = downstream_gate_H(Qb)

            if not np.isfinite(Hb) or Hb <= 0.0:
                return np.nan

            return Qb / (p.B * Hb) + 2.0 * np.sqrt(p.g * Hb) - R_plus_int

        q_scale = max(abs(QL), abs(p.Q_bar), 1.0)
        q_scale = min(q_scale, self.max_state_value)

        Q_lo = -10.0 * q_scale
        Q_hi = 10.0 * q_scale

        Q_right = None

        for _ in range(20):
            f_lo = downstream_eq(Q_lo)
            f_hi = downstream_eq(Q_hi)

            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0.0:
                Q_right = _bisect_root(downstream_eq, Q_lo, Q_hi)
                break

            Q_lo *= 2.0
            Q_hi *= 2.0

            if abs(Q_lo) > self.max_state_value or abs(Q_hi) > self.max_state_value:
                break

        if Q_right is None or not np.isfinite(Q_right):
            Q_right = QL

        H_right = downstream_gate_H(Q_right)

        if not np.isfinite(H_right):
            H_right = HL

        right_state = np.array([H_right, Q_right], dtype=np.float64)

        return left_state, right_state

    # ========================================================
    # Gym API
    # ========================================================

    def step(self, control):
        dx = self.dx
        dt_base = self.dt

        sample_rate = max(1, int(round(self.control_sample_rate / dt_base)))

        i = 0

        while i < sample_rate and self.time_index < self.nt - 1:
            if self._bad_state():
                break

            dt = min(dt_base, self.T - self.time)

            # CFL check. If violated, terminate this rollout.
            if not self._cfl_ok(dt):
                self._force_truncate = True
                break

            left_state, right_state = self._build_boundary_states(control)

            states = np.column_stack([self.H, self.Q])
            ext = np.vstack([left_state, states, right_state])

            try:
                fhat = rusanov_flux(ext[:-1], ext[1:], self.params)
                src = source(self.H, self.Q, self.params)

                if self.well_balanced:
                    states_new = (
                        states
                        - (dt / dx)
                        * (
                            (fhat[1:] - fhat[:-1])
                            - (self.star_flux[1:] - self.star_flux[:-1])
                        )
                        + dt * (src - self.star_src)
                    )
                else:
                    states_new = states - (dt / dx) * (fhat[1:] - fhat[:-1]) + dt * src

            except (FloatingPointError, OverflowError, ValueError):
                states_new = np.full_like(states, np.nan)

            # No clipping of H.
            self.H = states_new[:, 0]
            self.Q = states_new[:, 1]

            self.time += dt
            self.time_index += 1

            if self.time_index >= self.u.shape[0]:
                self._force_truncate = True
                break

            self.u[self.time_index] = self._state_vector()

            i += 1

            if self._bad_state():
                break

            if self.time >= self.T - 1e-12:
                break

        terminate = self.terminate()
        truncate = self.truncate()

        obs = self._get_obs()

        if self.reward_class is None:
            reward = 0.0
        else:
            try:
                reward = self.reward_class.reward(
                    self.u,
                    self.time_index,
                    terminate,
                    truncate,
                    self.last_control,
                    control_sample_rate=self.control_sample_rate,
                )
            except TypeError:
                reward = self.reward_class.reward(
                    self.u,
                    self.time_index,
                    terminate,
                    truncate,
                    self.last_control,
                )

        info = self.get_info()
        info["is_bad_state"] = bool(self._bad_state())
        info["terminated"] = bool(terminate)
        info["truncated"] = bool(truncate)

        return obs, reward, terminate, truncate, info

    def terminate(self):
        if self.time_index >= self.nt - 1 or self.time >= self.T - 1e-12:
            return True
        return False

    def truncate(self):
        return self._bad_state()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._force_truncate = False

        if self.reset_init_condition_func is not None:
            try:
                init_H, init_Q = self.reset_init_condition_func(self.nx)
            except TypeError:
                init_H, init_Q = self.reset_init_condition_func(
                    self.X_grid,
                    self.Hbar,
                    self.Qbar,
                    self.np_random,
                )

            self.H = np.asarray(init_H, dtype=np.float64)
            self.Q = np.asarray(init_Q, dtype=np.float64)

        else:
            self.H = np.asarray(self.H_init.copy(), dtype=np.float64)
            self.Q = np.asarray(self.Q_init.copy(), dtype=np.float64)

        self.u = np.zeros(
            (self.nt, 2 * (self.nx + 1)),
            dtype=np.float64,
        )

        self.u[0] = self._state_vector()

        self.time_index = 0
        self.time = 0.0
        self.last_control = self.params.U_L_bar

        return self._get_obs(), {}

    # ========================================================
    # Utilities
    # ========================================================

    def get_current_HQ(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.H.copy(), self.Q.copy()

    def get_info(self) -> Dict[str, float]:
        s = self.X_grid / self.params.L

        H_err = self.H - self.Hbar
        Q_err = self.Q - self.Qbar

        return {
            "time": float(self.time),
            "time_index": int(self.time_index),
            "H0": float(self.H[0]),
            "HL": float(self.H[-1]),
            "Q0": float(self.Q[0]),
            "QL": float(self.Q[-1]),
            "UL": float(self.last_control),
            "L2_H_error": float(np.sqrt(np.trapz(H_err**2, s))),
            "L2_Q_error": float(np.sqrt(np.trapz(Q_err**2, s))),
            "max_abs_H_error": float(np.max(np.abs(H_err))),
            "max_abs_Q_error": float(np.max(np.abs(Q_err))),
            "L2_H_state": float(np.sqrt(np.trapz(self.H * self.H, self.X_grid))),
            "L2_Q_state": float(np.sqrt(np.trapz(self.Q * self.Q, self.X_grid))),
        }