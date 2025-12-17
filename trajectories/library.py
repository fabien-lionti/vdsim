import numpy as np
from .base import BaseTrajectory, TrajectoryPoint

class StraightTrajectory(BaseTrajectory):
    def __init__(self, x0=0.0, y0=0.0, psi=0.0, v_ref=20.0):
        """
        Ligne droite infinie à vitesse constante.

        Args:
            x0, y0: position initiale
            psi: direction de la ligne (0 = axe X, pi/2 = axe Y)
            v_ref: vitesse de référence [m/s]
        """
        self.x0 = x0
        self.y0 = y0
        self.psi = psi
        self.v_ref = v_ref

    def sample(self, t: float) -> TrajectoryPoint:
        s = self.v_ref * t
        x = self.x0 + s * np.cos(self.psi)
        y = self.y0 + s * np.sin(self.psi)
        return TrajectoryPoint(
            t=t,
            x=x,
            y=y,
            psi=self.psi,
            v=self.v_ref,
            kappa=0.0,
        )

class SlalomTrajectory(BaseTrajectory):
    def __init__(self, v_ref, L=250.0, A=2.0, n_waves=3, L_straight=20.0, T_ramp=1.0):
        """
        Ligne droite + slalom avec entrée progressive (rampe sur l'amplitude).

        Args:
            v_ref: vitesse [m/s]
            L: longueur slalom [m]
            A: amplitude [m]
            n_waves: nb oscillations sur la durée du slalom
            L_straight: longueur ligne droite initiale [m]
            T_ramp: durée de montée progressive de l'amplitude [s] (C1)
        """
        self.v_ref = float(v_ref)
        self.L = float(L)
        self.A = float(A)
        self.n_waves = int(n_waves)
        self.L_straight = float(L_straight)
        self.T_ramp = float(T_ramp)

        self.T_straight = self.L_straight / self.v_ref
        self.T_slalom = self.L / self.v_ref
        self.T_total = self.T_straight + self.T_slalom

        self.omega = 2.0 * np.pi * self.n_waves / self.T_slalom

    @staticmethod
    def _smoothstep(u: float) -> float:
        # u in [0,1] -> s in [0,1], C1, pente nulle au début/fin
        u = np.clip(u, 0.0, 1.0)
        return 3.0*u*u - 2.0*u*u*u

    @staticmethod
    def _smoothstep_deriv(u: float) -> float:
        # dérivée par rapport à u
        u = np.clip(u, 0.0, 1.0)
        return 6.0*u - 6.0*u*u

    def sample(self, t: float) -> TrajectoryPoint:
        # Phase 0 : ligne droite
        if t <= self.T_straight:
            x = self.v_ref * t
            y = 0.0
            dx_dt = self.v_ref
            dy_dt = 0.0
            psi = 0.0
            v = self.v_ref
            kappa = 0.0
            return TrajectoryPoint(t=t, x=x, y=y, psi=psi, v=v, kappa=kappa)

        # Phase 1 : slalom (temps local)
        ts = t - self.T_straight
        ts = np.clip(ts, 0.0, self.T_slalom)

        # enveloppe d'amplitude progressive
        if self.T_ramp > 1e-9:
            u = ts / self.T_ramp
            s = self._smoothstep(u)
            ds_du = self._smoothstep_deriv(u)
            ds_dt = ds_du / self.T_ramp  # chaîne: ds/dt = (ds/du)*(du/dt)
        else:
            s = 1.0
            ds_dt = 0.0

        sin_ = np.sin(self.omega * ts)
        cos_ = np.cos(self.omega * ts)

        # position
        x = self.L_straight + self.v_ref * ts
        y = (self.A * s) * sin_

        # dérivées
        dx_dt = self.v_ref
        dy_dt = self.A * (ds_dt * sin_ + s * self.omega * cos_)

        psi = np.arctan2(dy_dt, dx_dt)
        v = self.v_ref

        # seconde dérivée de y (pour kappa)
        # y = A*s*sin(ωt)
        # y' = A*(s'*sin + s*ω*cos)
        # y'' = A*(s''*sin + 2*s'*ω*cos - s*ω^2*sin)
        if self.T_ramp > 1e-9:
            # s''(u) = 6 - 12u ; donc s''(t) = s''(u)/T_ramp^2
            u = np.clip(ts / self.T_ramp, 0.0, 1.0)
            d2s_du2 = 6.0 - 12.0*u
            d2s_dt2 = d2s_du2 / (self.T_ramp**2)
        else:
            d2s_dt2 = 0.0

        d2y_dt2 = self.A * (d2s_dt2 * sin_ + 2.0 * ds_dt * self.omega * cos_ - s * (self.omega**2) * sin_)

        # courbure approx (x'' = 0)
        speed_path = np.hypot(dx_dt, dy_dt)
        kappa = (dx_dt * d2y_dt2) / (speed_path**3 + 1e-9)

        return TrajectoryPoint(t=t, x=x, y=y, psi=psi, v=v, kappa=kappa)


class DoubleLaneChangeTrajectory(BaseTrajectory):
    def __init__(
        self,
        v_ref: float = 20.0,
        A: float = 3.5,
        L_total: float = 120.0,  # longueur de la manœuvre en X
        x0: float = 0.0,
        y0: float = 0.0,
    ):
        self.v_ref = v_ref
        self.A = A
        self.L_total = L_total
        self.x0 = x0
        self.y0 = y0

        # on découpe en 3 segments de même longueur
        self.L1 = L_total / 3.0
        self.L2 = 2.0 * L_total / 3.0
        self.L3 = L_total

    @staticmethod
    def _smooth_step_0_1(tau: float) -> float:
        return 0.5 - 0.5 * np.cos(np.pi * np.clip(tau, 0.0, 1.0))

    def _y_profile_s(self, s: float) -> float:
        """Profil latéral en fonction de la distance parcourue s."""
        A = self.A
        L1, L2, L3 = self.L1, self.L2, self.L3

        if s <= 0.0:
            return self.y0

        if s < L1:
            # 0 -> +A
            tau = s / L1
            return self.y0 + A * self._smooth_step_0_1(tau)

        elif s < L2:
            # +A -> -A
            tau = (s - L1) / (L2 - L1)
            return self.y0 + A - 2.0 * A * self._smooth_step_0_1(tau)

        elif s < L3:
            # -A -> 0
            tau = (s - L2) / (L3 - L2)
            return self.y0 - A + A * self._smooth_step_0_1(tau)

        else:
            return self.y0

    def sample(self, t: float) -> TrajectoryPoint:
        # progression le long de la trajectoire
        s = max(self.v_ref * t, 0.0)
        s = min(s, self.L_total)

        x = self.x0 + s
        y = self._y_profile_s(s)

        # dérivée par rapport à s pour calculer psi, v, kappa
        ds = 1e-3
        y_forward = self._y_profile_s(min(s + ds, self.L_total))
        dy_ds = (y_forward - y) / ds

        dx_ds = 1.0  # x = x0 + s -> dx/ds = 1

        # vitesse le long de la traj (≈ v_ref en module)
        ds_dt = self.v_ref
        dx_dt = dx_ds * ds_dt
        dy_dt = dy_ds * ds_dt

        psi = np.arctan2(dy_dt, dx_dt)
        v = np.hypot(dx_dt, dy_dt)

        # courbure
        y_backward = self._y_profile_s(max(s - ds, 0.0))
        dy_ds_back = (y - y_backward) / ds
        d2y_ds2 = (dy_ds - dy_ds_back) / ds

        # kappa(s) ≈ y'' / (1 + y'^2)^(3/2), mais ici dx_ds=1 => v_s^3≈(1+dy_ds^2)^(3/2)
        denom = (1.0 + dy_ds**2) ** 1.5 + 1e-9
        kappa = d2y_ds2 / denom

        return TrajectoryPoint(
            t=t,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )


class CircleTrajectory(BaseTrajectory):
    """
    Trajectoire circulaire à vitesse à peu près constante.
    Le véhicule tourne autour du centre (x_c, y_c) avec un rayon R.
    """

    def __init__(
        self,
        v_ref: float = 10.0,
        R: float = 20.0,
        x_c: float = 0.0,
        y_c: float = 0.0,
        clockwise: bool = False,
    ):
        """
        Args:
            v_ref: vitesse "cible" le long de la trajectoire [m/s]
            R: rayon du cercle [m]
            x_c, y_c: centre du cercle [m]
            clockwise: True pour tourner dans le sens horaire
        """
        self.v_ref = v_ref
        self.R = R
        self.x_c = x_c
        self.y_c = y_c

        # vitesse angulaire (signée)
        sgn = -1.0 if clockwise else 1.0
        self.omega = sgn * v_ref / R

    def _xy(self, t: float):
        theta = self.omega * t
        x = self.x_c + self.R * np.cos(theta)
        y = self.y_c + self.R * np.sin(theta)
        return x, y

    def sample(self, t: float) -> TrajectoryPoint:
        # Position
        x, y = self._xy(max(t, 0.0))

        # Dérivées numériques comme dans DoubleLaneChangeTrajectory
        dt = 1e-3
        x_f, y_f = self._xy(t + dt)
        x_b, y_b = self._xy(max(t - dt, 0.0))

        dx_dt = (x_f - x) / dt
        dy_dt = (y_f - y) / dt
        dx_dt_back = (x - x_b) / dt
        dy_dt_back = (y - y_b) / dt

        # orientation
        psi = np.arctan2(dy_dt, dx_dt)

        # vitesse
        v = np.hypot(dx_dt, dy_dt)

        # courbure numérique (comme pour le DLC)
        d2x_dt2 = (dx_dt - dx_dt_back) / dt
        d2y_dt2 = (dy_dt - dy_dt_back) / dt

        kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (v**3 + 1e-9)

        return TrajectoryPoint(
            t=t,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )
    
class CircleTrajectory(BaseTrajectory):
    """
    Cercle avec une ligne droite d'approche.
    - Phase 0: ligne droite alignée avec la tangente d'entrée du cercle
    - Phase 1: cercle
    """

    def __init__(
        self,
        v_ref: float = 10.0,
        R: float = 20.0,
        x_c: float = 0.0,
        y_c: float = 0.0,
        clockwise: bool = False,
        L_straight: float = 20.0,
    ):
        self.v_ref = float(v_ref)
        self.R = float(R)
        self.x_c = float(x_c)
        self.y_c = float(y_c)
        self.clockwise = bool(clockwise)

        self.L_straight = float(L_straight)
        self.T_straight = self.L_straight / self.v_ref if self.v_ref > 1e-9 else 0.0

        # vitesse angulaire signée
        sgn = -1.0 if self.clockwise else 1.0
        self.omega = sgn * self.v_ref / self.R

        # point d'entrée du cercle (theta=0)
        self.x_entry = self.x_c + self.R
        self.y_entry = self.y_c

        # direction tangente au point d'entrée
        # anti-horaire: tangente vers +y ; horaire: vers -y
        self.tan_dir = np.array([0.0, 1.0 if not self.clockwise else -1.0])

    def _xy_circle(self, t_circle: float):
        theta = self.omega * t_circle
        x = self.x_c + self.R * np.cos(theta)
        y = self.y_c + self.R * np.sin(theta)
        return x, y

    def _xy(self, t: float):
        t_eff = max(t, 0.0)

        # Phase 0 : ligne droite
        if t_eff <= self.T_straight:
            # on recule depuis le point d'entrée le long de la tangente
            s = self.v_ref * t_eff  # distance parcourue sur la ligne
            x = self.x_entry - self.tan_dir[0] * (self.L_straight - s)
            y = self.y_entry - self.tan_dir[1] * (self.L_straight - s)
            return x, y

        # Phase 1 : cercle
        t_circle = t_eff - self.T_straight
        return self._xy_circle(t_circle)

    def sample(self, t: float) -> TrajectoryPoint:
        t_eff = max(t, 0.0)

        # Position
        x, y = self._xy(t_eff)

        # Dérivées numériques
        dt = 1e-3
        x_f, y_f = self._xy(t_eff + dt)
        x_b, y_b = self._xy(max(t_eff - dt, 0.0))

        dx_dt = (x_f - x) / dt
        dy_dt = (y_f - y) / dt
        dx_dt_back = (x - x_b) / dt
        dy_dt_back = (y - y_b) / dt

        psi = np.arctan2(dy_dt, dx_dt)
        v = np.hypot(dx_dt, dy_dt)

        d2x_dt2 = (dx_dt - dx_dt_back) / dt
        d2y_dt2 = (dy_dt - dy_dt_back) / dt

        kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (v**3 + 1e-9)

        return TrajectoryPoint(
            t=t_eff,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )


class LemniscateTrajectory(BaseTrajectory):
    """
    Trajectoire en lemniscate (∞) type Lissajous.
    """

    def __init__(
        self,
        v_ref: float = 10.0,
        A: float = 10.0,
        B: float = 10,
        T_period: float = 10.0,
        x0: float = 0.0,
        y0: float = 0.0,
    ):
        """
        Args:
            v_ref: juste informatif ici (la vitesse varie légèrement)
            A: amplitude principale en x [m]
            B: amplitude en y [m] (par défaut A/2 pour un joli ∞)
            T_period: période de la figure (temps pour refaire un 8 complet) [s]
            x0, y0: centre de la lemniscate [m]
        """
        self.v_ref = v_ref
        self.A = A
        self.B = B if B is not None else A / 2.0
        self.T_period = T_period
        self.x0 = x0
        self.y0 = y0

        # fréquence angulaire
        self.omega = 2.0 * np.pi / T_period

    def _xy(self, t: float):
        w_t = self.omega * t
        x = self.x0 + self.A * np.sin(w_t)
        y = self.y0 + self.B * np.sin(2.0 * w_t)
        return x, y

    def sample(self, t: float) -> TrajectoryPoint:
        t_eff = max(t, 0.0)
        x, y = self._xy(t_eff)

        # Dérivées numériques
        dt = 1e-3
        x_f, y_f = self._xy(t_eff + dt)
        x_b, y_b = self._xy(max(t_eff - dt, 0.0))

        dx_dt = (x_f - x) / dt
        dy_dt = (y_f - y) / dt
        dx_dt_back = (x - x_b) / dt
        dy_dt_back = (y - y_b) / dt

        psi = np.arctan2(dy_dt, dx_dt)
        v = np.hypot(dx_dt, dy_dt)

        d2x_dt2 = (dx_dt - dx_dt_back) / dt
        d2y_dt2 = (dy_dt - dy_dt_back) / dt

        kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (v**3 + 1e-9)

        return TrajectoryPoint(
            t=t_eff,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )

class WaypointTrajectory(BaseTrajectory):
    """
    Trajectoire définie par une liste de waypoints (t, x, y),
    avec interpolation linéaire entre les points.
    """

    def __init__(self, t_points, x_points, y_points):
        """
        Args:
            t_points, x_points, y_points: listes/arrays de même taille,
                t_points strictement croissants.
        """
        self.t = np.asarray(t_points)
        self.x = np.asarray(x_points)
        self.y = np.asarray(y_points)
        assert self.t.shape == self.x.shape == self.y.shape
        assert np.all(np.diff(self.t) > 0.0)

    def _interp_xy(self, t: float):
        t_clamped = np.clip(t, self.t[0], self.t[-1])
        x = np.interp(t_clamped, self.t, self.x)
        y = np.interp(t_clamped, self.t, self.y)
        return x, y

    def sample(self, t: float) -> TrajectoryPoint:
        t_eff = t

        x, y = self._interp_xy(t_eff)

        # dérivées numériques
        dt = 1e-3
        x_f, y_f = self._interp_xy(t_eff + dt)
        x_b, y_b = self._interp_xy(t_eff - dt)

        dx_dt = (x_f - x) / dt
        dy_dt = (y_f - y) / dt
        dx_dt_back = (x - x_b) / dt
        dy_dt_back = (y - y_b) / dt

        psi = np.arctan2(dy_dt, dx_dt)
        v = np.hypot(dx_dt, dy_dt)

        d2x_dt2 = (dx_dt - dx_dt_back) / dt
        d2y_dt2 = (dy_dt - dy_dt_back) / dt
        kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (v**3 + 1e-9)

        return TrajectoryPoint(
            t=t_eff,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )

# class LemniscateTrajectory(BaseTrajectory):
#     """
#     Lemniscate (∞) type Lissajous + ligne droite initiale.
#     Géométrie indépendante de v_ref.
#     """

#     def __init__(
#         self,
#         v_ref: float = 10.0,      # utilisé comme consigne de vitesse (pas pour la géométrie)
#         A: float = 10.0,
#         B: float = 10.0,
#         T_period: float = 30.0,   # définit la "vitesse géométrique" de la figure (indépendant de v_ref)
#         x0: float = 0.0,
#         y0: float = 0.0,
#         L_straight: float = 20.0, # longueur ligne droite initiale
#     ):
#         """
#         Args:
#             v_ref: vitesse de référence VEHICULE [m/s] (utilisée dans TrajectoryPoint.v)
#             A: amplitude en x [m]
#             B: amplitude en y [m]
#             T_period: période géométrique de la figure (temps pour un 8 complet) [s]
#             x0, y0: centre de la figure [m]
#             L_straight: longueur ligne droite avant d'entrer dans le 8 [m]
#         """
#         self.v_ref = float(v_ref)
#         self.A = float(A)
#         self.B = float(B) if B is not None else float(A) / 2.0
#         self.T_period = float(T_period)
#         self.x0 = float(x0)
#         self.y0 = float(y0)

#         self.L_straight = float(L_straight)
#         self.T_straight = self.L_straight / self.v_ref if self.v_ref > 1e-9 else 0.0

#         # fréquence angulaire (géométrie)
#         self.omega = 2.0 * np.pi / self.T_period

#         # point d'entrée de la lemniscate (t_local=0)
#         self.x_entry, self.y_entry = self._xy_lem(0.0)

#         # direction tangente au départ de la lemniscate (pour aligner la ligne droite)
#         dx0, dy0 = self._dxy_dt_lem(0.0)
#         n = np.hypot(dx0, dy0) + 1e-12
#         self.tan_dir = np.array([dx0 / n, dy0 / n])  # unitaire

#         # point de départ de la ligne droite (en amont)
#         self.x_start = self.x_entry - self.tan_dir[0] * self.L_straight
#         self.y_start = self.y_entry - self.tan_dir[1] * self.L_straight

#         # (optionnel) warning de cohérence: vitesse moyenne géométrique
#         # approx longueur du trajet / période (grossier, mais utile)
#         # Tu peux le retirer si tu veux.
#         # self.v_geom ~ amplitude * omega (ordre de grandeur)
#         v_geom = self.A * self.omega
#         if abs(v_geom - self.v_ref) / max(self.v_ref, 1e-6) > 0.5:
#             print(
#                 f"[LemniscateTrajectory] Warning: v_ref={self.v_ref:.2f} m/s "
#                 f"peut être très différent de l'échelle géométrique (~{v_geom:.2f} m/s)."
#             )

#     # ---------- Lemniscate pure (temps local) ----------
#     def _xy_lem(self, t_local: float):
#         w_t = self.omega * t_local
#         x = self.x0 + self.A * np.sin(w_t)
#         y = self.y0 + self.B * np.sin(2.0 * w_t)
#         return x, y

#     def _dxy_dt_lem(self, t_local: float):
#         # dérivées analytiques (mieux que numérique pour la tangente)
#         w_t = self.omega * t_local
#         dx_dt = self.A * self.omega * np.cos(w_t)
#         dy_dt = 2.0 * self.B * self.omega * np.cos(2.0 * w_t)
#         return dx_dt, dy_dt

#     # ---------- Trajectoire complète (ligne + lemniscate) ----------
#     def _xy(self, t: float):
#         t_eff = max(t, 0.0)

#         # Phase 0 : ligne droite jusqu'au point d'entrée
#         if t_eff <= self.T_straight:
#             # progression le long de la tangente
#             s = self.v_ref * t_eff
#             x = self.x_start + self.tan_dir[0] * s
#             y = self.y_start + self.tan_dir[1] * s
#             return x, y

#         # Phase 1 : lemniscate (temps local décalé)
#         t_local = t_eff - self.T_straight
#         return self._xy_lem(t_local)

#     def sample(self, t: float) -> TrajectoryPoint:
#         t_eff = max(t, 0.0)
#         x, y = self._xy(t_eff)

#         # Dérivées numériques (comme ton style)
#         dt = 1e-3
#         x_f, y_f = self._xy(t_eff + dt)
#         x_b, y_b = self._xy(max(t_eff - dt, 0.0))

#         dx_dt = (x_f - x) / dt
#         dy_dt = (y_f - y) / dt
#         dx_dt_back = (x - x_b) / dt
#         dy_dt_back = (y - y_b) / dt

#         psi = np.arctan2(dy_dt, dx_dt)

#         # vitesse de référence (pour le contrôleur)
#         v = self.v_ref

#         d2x_dt2 = (dx_dt - dx_dt_back) / dt
#         d2y_dt2 = (dy_dt - dy_dt_back) / dt
#         speed_path = np.hypot(dx_dt, dy_dt)
#         kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed_path**3 + 1e-9)

#         return TrajectoryPoint(
#             t=t_eff,
#             x=x,
#             y=y,
#             psi=psi,
#             v=v,
#             kappa=kappa,
#         )

import numpy as np

# suppose que BaseTrajectory et TrajectoryPoint existent déjà dans ton projet
# from your_module import BaseTrajectory, TrajectoryPoint


class LemniscateTrajectory(BaseTrajectory):
    """
    Lemniscate (∞) type Lissajous + ligne droite initiale.

    IMPORTANT:
    - Ici, v_ref impose la loi temporelle:
      au temps t, la position correspond à s(t)=v_ref*t le long du chemin.
    - La lemniscate est donc reparamétrée par l'abscisse curviligne (arc-length).
    """

    def __init__(
        self,
        v_ref: float = 10.0,      # vitesse véhicule [m/s] => dicte la timeline
        A: float = 10.0,
        B: float = 10.0,
        T_period: float = 30.0,   # (optionnel ici) sert juste à régler la résolution interne si tu veux
        x0: float = 0.0,
        y0: float = 0.0,
        L_straight: float = 20.0, # longueur ligne droite initiale
        n_table: int = 4000,      # résolution table arc-length
    ):
        self.v_ref = float(v_ref)
        self.A = float(A)
        self.B = float(B) if B is not None else float(A) / 2.0
        self.T_period = float(T_period)  # gardé pour compat/usage futur
        self.x0 = float(x0)
        self.y0 = float(y0)

        self.L_straight = float(L_straight)
        self.T_straight = self.L_straight / self.v_ref if self.v_ref > 1e-9 else 0.0

        # --------- Table s(φ) pour reparamétrer par l'arc-length ----------
        # Paramètre géométrique: φ ∈ [0, 2π] (un "8" complet)
        self.n_table = int(max(200, n_table))
        self.phi_grid = np.linspace(0.0, 2.0 * np.pi, self.n_table)

        # x(φ), y(φ)
        xg, yg = self._xy_phi(self.phi_grid)

        # dérivées par rapport à φ: dx/dφ, dy/dφ
        dx_dphi, dy_dphi = self._dxy_dphi(self.phi_grid)

        # ds/dφ = ||dp/dφ||
        ds_dphi = np.hypot(dx_dphi, dy_dphi)

        # intégration trapézoïdale pour obtenir s(φ)
        dphi = self.phi_grid[1] - self.phi_grid[0]
        # cumtrapz "maison" (évite scipy): s[0]=0
        self.s_grid = np.zeros_like(self.phi_grid)
        self.s_grid[1:] = np.cumsum(0.5 * (ds_dphi[1:] + ds_dphi[:-1]) * dphi)

        self.L_period = float(self.s_grid[-1])  # longueur d'un 8 complet

        if self.L_period < 1e-9:
            raise ValueError("Longueur de trajectoire (L_period) ~ 0. Vérifie A, B.")

        # --------- Début lemniscate (φ=0) ----------
        self.x_entry, self.y_entry = self._xy_phi(0.0)

        # direction tangente (unitaire) au départ de la lemniscate (φ=0)
        dx0, dy0 = self._dxy_dphi(0.0)
        n = np.hypot(dx0, dy0) + 1e-12
        self.tan_dir = np.array([dx0 / n, dy0 / n])

        # point de départ de la ligne droite (en amont)
        self.x_start = self.x_entry - self.tan_dir[0] * self.L_straight
        self.y_start = self.y_entry - self.tan_dir[1] * self.L_straight

    # ---------- Géométrie lemniscate: paramètre φ ----------
    def _xy_phi(self, phi):
        # x = x0 + A sin(φ), y = y0 + B sin(2φ)
        x = self.x0 + self.A * np.sin(phi)
        y = self.y0 + self.B * np.sin(2.0 * phi)
        return x, y

    def _dxy_dphi(self, phi):
        # dx/dφ = A cos(φ)
        # dy/dφ = 2B cos(2φ)
        dx = self.A * np.cos(phi)
        dy = 2.0 * self.B * np.cos(2.0 * phi)
        return dx, dy

    # ---------- Mapping s -> φ via interpolation ----------
    def _phi_from_s(self, s):
        # s dans [0, L_period)
        s_mod = np.mod(s, self.L_period)
        # interpolation monotone: s_grid -> phi_grid
        return np.interp(s_mod, self.s_grid, self.phi_grid)

    # ---------- Lemniscate "time-based" avec v_ref: s(t)=v_ref*t ----------
    def _xy_lem(self, t_local: float):
        s = self.v_ref * max(t_local, 0.0)
        phi = self._phi_from_s(s)
        return self._xy_phi(phi)

    # ---------- Trajectoire complète (ligne + lemniscate) ----------
    def _xy(self, t: float):
        t_eff = max(t, 0.0)

        # Phase 0 : ligne droite jusqu'au point d'entrée (à vitesse v_ref)
        if t_eff <= self.T_straight:
            s = self.v_ref * t_eff
            x = self.x_start + self.tan_dir[0] * s
            y = self.y_start + self.tan_dir[1] * s
            return x, y

        # Phase 1 : lemniscate, en "arc-length time law"
        t_local = t_eff - self.T_straight
        return self._xy_lem(t_local)

    def sample(self, t: float):
        t_eff = max(t, 0.0)
        x, y = self._xy(t_eff)

        # Dérivées numériques (centrées) + accélération centrée
        dt = 1e-3
        x_f, y_f = self._xy(t_eff + dt)
        x_b, y_b = self._xy(max(t_eff - dt, 0.0))

        # vitesse (centrée)
        dx_dt = (x_f - x_b) / (2.0 * dt)
        dy_dt = (y_f - y_b) / (2.0 * dt)

        # accélération (centrée)
        d2x_dt2 = (x_f - 2.0 * x + x_b) / (dt * dt)
        d2y_dt2 = (y_f - 2.0 * y + y_b) / (dt * dt)

        psi = np.arctan2(dy_dt, dx_dt)

        speed_path = np.hypot(dx_dt, dy_dt)
        kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed_path**3 + 1e-9)

        # Ici, v_ref est bien la "timeline": à t, on vise la position correspondant à s=v_ref*t
        v = self.v_ref

        return TrajectoryPoint(
            t=t_eff,
            x=x,
            y=y,
            psi=psi,
            v=v,
            kappa=kappa,
        )

