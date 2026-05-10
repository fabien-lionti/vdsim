# trajectories/smooth_random.py
"""
Trajectoire aléatoire réaliste avec spline cubique C2 et vitesse variable.

- Waypoints aléatoires avec x monotone croissant
- Spline cubique paramétrique (C2 continu)
- Reparamétrisation par abscisse curviligne (arc-length)
- Profil de vitesse variable: décélération en virage (freinage + virage combiné)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from .base import BaseTrajectory, TrajectoryPoint


class SmoothRandomTrajectory(BaseTrajectory):
    """
    Trajectoire aléatoire lisse (C2) avec profil de vitesse variable.

    Le véhicule ralentit dans les virages et accélère en ligne droite,
    créant naturellement des manœuvres combinées freinage + virage.
    """

    def __init__(
        self,
        v_ref: float = 15.0,
        n_waypoints: int = 8,
        d_min: float = 50.0,
        y_max: float = 5.0,
        alpha: float = 0.5,
        v_min_ratio: float = 0.5,
        kappa_ref: float = 0.02,
        L_straight: float = 80.0,
        seed: int = 0,
        n_table: int = 4000,
        kappa_max_reject: float = 0.15,
        max_retries: int = 50,
    ):
        """
        Args:
            v_ref: vitesse de référence [m/s]
            n_waypoints: nombre de waypoints (5-12)
            d_min: espacement minimum entre waypoints en x [m]
            y_max: déviation latérale max [m]
            alpha: agressivité de la décélération en virage (0.3-0.8)
            v_min_ratio: plancher de vitesse relatif (0.3-0.7)
            kappa_ref: courbure de référence pour le profil de vitesse [1/m]
            L_straight: longueur du segment droit initial [m]
            seed: seed pour reproductibilité
            n_table: résolution de la table arc-length
            kappa_max_reject: seuil de courbure max (rejeter si dépassé)
            max_retries: tentatives max pour générer des waypoints valides
        """
        self.v_ref = float(v_ref)
        self.n_waypoints = int(n_waypoints)
        self.d_min = float(d_min)
        self.y_max = float(y_max)
        self.alpha = float(alpha)
        self.v_min_ratio = float(v_min_ratio)
        self.kappa_ref = float(kappa_ref)
        self.L_straight = float(L_straight)
        self.n_table = int(max(500, n_table))
        self.kappa_max_reject = float(kappa_max_reject)

        self.rng = np.random.RandomState(seed)

        # --- A. Générer les waypoints ---
        wx, wy = self._generate_waypoints(max_retries)

        # --- B. Spline cubique paramétrique ---
        # Paramètre u ∈ [0, 1] uniforme sur les waypoints
        # Clamped BC: tangente au départ alignée avec l'axe x (raccord C1 avec la ligne droite)
        n_wp = len(wx)
        u_wp = np.linspace(0.0, 1.0, n_wp)
        # Tangente imposée au départ: direction x pure (scale ~ longueur totale en x)
        dx_total = wx[-1] - wx[0]
        self.spline_x = CubicSpline(u_wp, wx, bc_type=((1, dx_total), (2, 0.0)))
        self.spline_y = CubicSpline(u_wp, wy, bc_type=((1, 0.0), (2, 0.0)))

        # --- C. Arc-length reparamétrisation ---
        self.u_grid = np.linspace(0.0, 1.0, self.n_table)
        dx_du = self.spline_x(self.u_grid, 1)  # première dérivée
        dy_du = self.spline_y(self.u_grid, 1)
        ds_du = np.hypot(dx_du, dy_du)

        du = self.u_grid[1] - self.u_grid[0]
        self.s_grid = np.zeros(self.n_table)
        self.s_grid[1:] = np.cumsum(0.5 * (ds_du[1:] + ds_du[:-1]) * du)
        self.L_spline = float(self.s_grid[-1])

        # Longueur totale (straight + spline)
        self.L_total = self.L_straight + self.L_spline
        self.T_straight = self.L_straight / self.v_ref if self.v_ref > 1e-9 else 0.0

        # --- D. Profil de vitesse variable v(s) ---
        # Calculer kappa sur la grille dense
        d2x_du2 = self.spline_x(self.u_grid, 2)
        d2y_du2 = self.spline_y(self.u_grid, 2)
        # kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numer = np.abs(dx_du * d2y_du2 - dy_du * d2x_du2)
        denom = (dx_du**2 + dy_du**2)**1.5 + 1e-12
        kappa_grid = numer / denom

        # Profil brut: v(s) = v_ref * clamp(1 - alpha * |kappa| / kappa_ref, v_min_ratio, 1)
        v_ratio = 1.0 - self.alpha * kappa_grid / self.kappa_ref
        v_ratio = np.clip(v_ratio, self.v_min_ratio, 1.0)

        # Lissage gaussien (sigma ~ 1% de la grille)
        sigma = max(1, self.n_table // 100)
        v_ratio_smooth = gaussian_filter1d(v_ratio, sigma=sigma)
        v_ratio_smooth = np.clip(v_ratio_smooth, self.v_min_ratio, 1.0)

        # Raccord C1 avec la phase droite: forcer v(s=0) = v_ref
        # et rampe smooth sur les premiers 5% de la spline
        n_ramp = max(1, self.n_table // 20)  # 5% de la grille
        ramp = np.linspace(0.0, 1.0, n_ramp)
        ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))  # smoothstep (cosine blend)
        v_ratio_smooth[:n_ramp] = 1.0 + ramp * (v_ratio_smooth[:n_ramp] - 1.0)
        # Idem en sortie pour éviter un arrêt brusque
        v_ratio_smooth[-n_ramp:] = v_ratio_smooth[-n_ramp:] + ramp[::-1] * (1.0 - v_ratio_smooth[-n_ramp:])
        v_ratio_smooth = np.clip(v_ratio_smooth, self.v_min_ratio, 1.0)

        self.v_grid = self.v_ref * v_ratio_smooth  # v(u) sur la grille

        # --- E. Loi temporelle t(s) par intégration de dt = ds / v(s) ---
        # On calcule t en fonction de s (sur la spline uniquement)
        ds_vals = np.diff(self.s_grid)
        v_mid = 0.5 * (self.v_grid[:-1] + self.v_grid[1:])
        v_mid = np.maximum(v_mid, 1e-3)  # éviter division par zéro
        dt_vals = ds_vals / v_mid

        self.t_spline_grid = np.zeros(self.n_table)
        self.t_spline_grid[1:] = np.cumsum(dt_vals)
        self.T_spline = float(self.t_spline_grid[-1])
        self.T_total = self.T_straight + self.T_spline

    def _generate_waypoints(self, max_retries: int):
        """Génère N waypoints aléatoires avec x monotone croissant."""
        for _ in range(max_retries):
            n = self.n_waypoints

            # Espacements aléatoires en x (>= d_min)
            dx = self.rng.uniform(self.d_min, self.d_min * 2.0, size=n - 1)
            wx = np.zeros(n)
            wx[1:] = np.cumsum(dx)

            # Déviations latérales (premier et dernier = 0 pour smooth entry/exit)
            wy = np.zeros(n)
            wy[1:-1] = self.rng.uniform(-self.y_max, self.y_max, size=n - 2)

            # Vérification: spline temporaire pour tester kappa_max
            u_wp = np.linspace(0.0, 1.0, n)
            dx_tot = wx[-1] - wx[0]
            spx = CubicSpline(u_wp, wx, bc_type=((1, dx_tot), (2, 0.0)))
            spy = CubicSpline(u_wp, wy, bc_type=((1, 0.0), (2, 0.0)))

            u_test = np.linspace(0.0, 1.0, 2000)
            dx_du = spx(u_test, 1)
            dy_du = spy(u_test, 1)
            d2x_du2 = spx(u_test, 2)
            d2y_du2 = spy(u_test, 2)

            numer = np.abs(dx_du * d2y_du2 - dy_du * d2x_du2)
            denom = (dx_du**2 + dy_du**2)**1.5 + 1e-12
            kappa_test = numer / denom

            if np.max(kappa_test) <= self.kappa_max_reject:
                return wx, wy

        # Si toutes les tentatives échouent, retourner la dernière (best effort)
        return wx, wy

    def _u_from_s(self, s: float) -> float:
        """Inverse: s → u via interpolation sur la table."""
        s_clamped = np.clip(s, 0.0, self.L_spline)
        return float(np.interp(s_clamped, self.s_grid, self.u_grid))

    def _s_from_t_spline(self, t_local: float) -> float:
        """Inverse: t_local → s via interpolation sur la table temporelle."""
        t_clamped = np.clip(t_local, 0.0, self.T_spline)
        return float(np.interp(t_clamped, self.t_spline_grid, self.s_grid))

    def _v_from_s(self, s: float) -> float:
        """Retourne v(s) par interpolation."""
        s_clamped = np.clip(s, 0.0, self.L_spline)
        return float(np.interp(s_clamped, self.s_grid, self.v_grid))

    def _xy_spline(self, u: float):
        """Position sur la spline pour le paramètre u."""
        return float(self.spline_x(u)), float(self.spline_y(u))

    def _xy(self, t: float):
        """Position (x, y) au temps t (straight + spline)."""
        t_eff = max(t, 0.0)

        if t_eff <= self.T_straight:
            # Phase 0: ligne droite initiale le long de l'axe x
            s = self.v_ref * t_eff
            x = -self.L_straight + s
            y = 0.0
            return x, y

        # Phase 1: spline avec vitesse variable
        t_local = t_eff - self.T_straight
        t_local = min(t_local, self.T_spline)

        s = self._s_from_t_spline(t_local)
        u = self._u_from_s(s)
        return self._xy_spline(u)

    def sample(self, t: float) -> TrajectoryPoint:
        t_eff = max(t, 0.0)
        x, y = self._xy(t_eff)

        # Dérivées numériques centrées (même pattern que les autres trajectoires)
        dt = 1e-3
        x_f, y_f = self._xy(t_eff + dt)
        x_b, y_b = self._xy(max(t_eff - dt, 0.0))

        dx_dt = (x_f - x_b) / (2.0 * dt)
        dy_dt = (y_f - y_b) / (2.0 * dt)

        d2x_dt2 = (x_f - 2.0 * x + x_b) / (dt * dt)
        d2y_dt2 = (y_f - 2.0 * y + y_b) / (dt * dt)

        psi = np.arctan2(dy_dt, dx_dt)
        speed_path = np.hypot(dx_dt, dy_dt)
        kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed_path**3 + 1e-9)

        # Vitesse de référence: constante pendant straight, variable pendant spline
        if t_eff <= self.T_straight:
            v = self.v_ref
        else:
            t_local = min(t_eff - self.T_straight, self.T_spline)
            s = self._s_from_t_spline(t_local)
            v = self._v_from_s(s)

        return TrajectoryPoint(
            t=t_eff,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )
