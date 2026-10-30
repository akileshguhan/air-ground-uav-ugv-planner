"""PCA feature extraction: batched KD-tree + stacked eigh (refactored version)."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from .compat import kdtree_query, kdtree_query_radius
from .protocol import SegmentFeatures


class PointCloudFeatureExtractor:
    def __init__(self, k_neighbors: int = 12, neighborhood_radius_m: Optional[float] = None,
                 roughness_scale: float = 20.0, n_sample: int = 120, n_jobs: int = -1,
                 planar_curvature_max: float = 0.02, min_planar_fraction: float = 0.2,
                 step_window_m: float = 0.2, step_min_points: int = 8,
                 step_max_neighbors: int = 256, step_trim_percent: float = 1.0,
                 gap_mode: str = "spacing", gap_track_offsets_m=(-0.25, 0.0, 0.25)):
        """planar_curvature_max: slope is measured only from neighbourhoods whose
        PCA curvature (smallest eigenvalue / sum) is below this, i.e. that lie on
        ONE surface. A neighbourhood straddling a step fits a single tilted plane
        across two levels and reports the step face as a steep incline (an 8 cm
        step read as ~34 deg). Steps are the step metric's job, not the slope
        metric's. Measured separation: planar p99 <= 0.0055 vs step edges >= 0.036
        at 2.5 m altitude, but only 0.026 vs 0.025 at 4 m -- noise-limited.
        UNCALIBRATED.
        min_planar_fraction: if fewer than this share of samples are planar, fall
        back to all samples rather than judge slope from a handful of points.
        Conservative: never lets a mostly-broken segment pass on thin evidence."""
        # Step metric (see _local_step_height): patch radius ~ wheel scale.
        # UNCALIBRATED -- should match the vehicle's wheel/footprint size.
        self.step_window_m = float(step_window_m)
        self.step_min_points = int(step_min_points)
        self.step_max_neighbors = int(step_max_neighbors)
        # Spread = (100 - t)th minus t-th percentile of patch residuals. Default
        # 1.0 is the behaviour verified on the D455 data. Raise it (e.g. 5) for
        # sparse or noisy clouds: a real step splits a patch roughly half above,
        # half below, so trimming the outer 5% barely moves it, while it drops
        # the one or two noisy cells that otherwise set the whole reading.
        self.step_trim_percent = float(step_trim_percent)
        # gap_mode "spacing" (default, verified on D455): largest spacing
        # between points sorted along the segment. On a coarse grid this reads
        # the grid spacing itself as a gap (0.5 m cells at 45 deg -> 0.35 m).
        # "holes": sample the centreline and wheel tracks, flag spots with no
        # ground point within 1.5 x the data's own spacing s, and report the
        # longest flagged run + 2s (its true width). Grid-orientation
        # independent; cannot resolve holes narrower than ~2s.
        if gap_mode not in ("spacing", "holes"):
            raise ValueError("gap_mode must be 'spacing' or 'holes'")
        self.gap_mode = gap_mode
        self.gap_track_offsets_m = tuple(gap_track_offsets_m)
        self.planar_curvature_max = float(planar_curvature_max)
        self.min_planar_fraction = float(min_planar_fraction)
        self.k = int(k_neighbors)
        self.radius = neighborhood_radius_m
        self.roughness_scale = float(roughness_scale)
        self.n_sample = int(n_sample)
        self.n_jobs = int(n_jobs)

    def _normals_and_curvature(self, pts, sample_idx):
        tree = cKDTree(pts)
        q = pts[sample_idx]
        if self.radius is None:
            return self._batched_fixed_k(pts, tree, q)
        return self._loop_fixed_radius(pts, tree, q)

    def _batched_fixed_k(self, pts, tree, q):
        k = min(self.k + 1, pts.shape[0])
        _, idx = kdtree_query(tree, q, k=k, n_jobs=self.n_jobs)
        if idx.ndim == 1:
            idx = idx[:, None]
        idx = idx[:, 1:] if idx.shape[1] > 1 else idx
        nb = pts[idx]
        d = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", d, d) / max(nb.shape[1] - 1, 1)
        w, v = np.linalg.eigh(cov)
        n = v[:, :, 0]
        flip = n[:, 2] < 0
        n[flip] = -n[flip]
        w = np.clip(w, 1e-12, None)
        return n, w[:, 0] / w.sum(axis=1)

    def _loop_fixed_radius(self, pts, tree, q):
        neigh = kdtree_query_radius(tree, q, self.radius, n_jobs=self.n_jobs)
        normals = np.zeros((len(q), 3)); curv = np.zeros(len(q))
        for i, ids in enumerate(neigh):
            ids = np.asarray(ids, dtype=int)
            if ids.size < 3:
                normals[i] = (0, 0, 1); continue
            nb = pts[ids]; d = nb - nb.mean(axis=0)
            w, v = np.linalg.eigh(d.T @ d / max(len(ids) - 1, 1))
            nn = v[:, 0]
            if nn[2] < 0:
                nn = -nn
            w = np.clip(w, 1e-12, None)
            normals[i] = nn; curv[i] = w[0] / w.sum()
        return normals, curv

    def _local_step_height(self, pts, tree) -> float:
        """Largest local height discontinuity, independent of slope direction.

        Replaces the old metric (sort points along the segment, take the largest
        consecutive height difference). That leaked the terrain's own slope into
        the reading whenever the segment ran at an angle to the slope: on a 12
        deg ramp a diagonal segment 2.2 m wide read 0.25 m of "step" from
        smooth ground (2.2 m x sin 31 deg x tan 12 deg = 0.24 m), and 16 ramp
        segments were falsely rejected.

        New metric: on a grid of patch centres (spacing = window radius, so any
        step line lies within 0.71 R of some centre), fit z = a*x + b*y + c to
        the points within radius R and take the spread of the residuals
        (99th - 1st percentile).
          - smooth surface at any slope: plane fits, spread ~ noise
          - step of height h: plane sits midway, residuals ~ +h/2 and -h/2,
            spread ~ h, regardless of the slope underneath
        Curvature leaks in as roughly kappa * R^2 / 2 (0.05 m at the bottom of
        a 43 deg dip); steep dips are caught by the slope metric anyway.
        Assumes 2.5D terrain (height as a function of x, y).
        """
        R = self.step_window_m
        lo = pts[:, :2].min(axis=0)
        hi = pts[:, :2].max(axis=0)
        gx = np.arange(lo[0], hi[0] + 1e-9, R)
        gy = np.arange(lo[1], hi[1] + 1e-9, R)
        cx, cy = np.meshgrid(gx, gy)
        centres = np.stack([cx.ravel(), cy.ravel()], axis=1)

        tree2d = cKDTree(pts[:, :2])
        neigh = kdtree_query_radius(tree2d, centres, R, n_jobs=self.n_jobs)
        # Build the padded neighbour table without a per-patch Python loop.
        lens = np.fromiter((len(x) for x in neigh), dtype=np.int64, count=len(neigh))
        keep = np.flatnonzero(lens >= self.step_min_points)
        if keep.size == 0:
            return 0.0
        flat = np.concatenate([np.asarray(neigh[i], dtype=np.int64) for i in keep])
        klen = lens[keep]
        starts = np.concatenate([[0], np.cumsum(klen)[:-1]])
        cap = self.step_max_neighbors
        use = np.minimum(klen, cap)                          # density-independent cost
        m = keep.size
        maxn = int(use.max())
        j = np.arange(maxn)[None, :]
        mask = j < use[:, None]
        # evenly strided pick within each row when a row exceeds the cap
        pos = np.where(klen[:, None] > cap,
                       (j * (klen[:, None] - 1)) // np.maximum(use[:, None] - 1, 1), j)
        idx = flat[np.minimum(starts[:, None] + pos, flat.size - 1)]
        idx = np.where(mask, idx, 0)

        P = pts[idx]                                       # (m, maxn, 3)
        ctr = np.where(mask[:, :, None], P, 0.0).sum(axis=1) / mask.sum(axis=1)[:, None]
        dx = (P[:, :, 0] - ctr[:, None, 0]) * mask         # centred for conditioning
        dy = (P[:, :, 1] - ctr[:, None, 1]) * mask
        z = P[:, :, 2] * mask
        w = mask.astype(float)

        A = np.empty((m, 3, 3))
        A[:, 0, 0] = (dx * dx).sum(1); A[:, 0, 1] = (dx * dy).sum(1); A[:, 0, 2] = dx.sum(1)
        A[:, 1, 1] = (dy * dy).sum(1); A[:, 1, 2] = dy.sum(1); A[:, 2, 2] = w.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9            # collinear patches stay solvable
        b = np.stack([(dx * z).sum(1), (dy * z).sum(1), z.sum(1)], axis=1)
        coef = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # a, b, c per patch

        res = P[:, :, 2] - (coef[:, 0:1] * dx + coef[:, 1:2] * dy + coef[:, 2:3])
        # Vectorised percentiles. np.nanpercentile loops over rows in Python and
        # took ~90% of feature time; sorting once (NaN sorts last) and indexing
        # the 1st/99th-percentile ranks per row is equivalent and ~20x faster.
        res = np.sort(np.where(mask, res, np.nan), axis=1)
        n = mask.sum(axis=1)
        t = self.step_trim_percent / 100.0
        lo = np.rint(t * (n - 1)).astype(int)
        hi = np.rint((1.0 - t) * (n - 1)).astype(int)
        rows_i = np.arange(m)
        spread = res[rows_i, hi] - res[rows_i, lo]
        return float(np.max(spread))

    def _hole_gap(self, pts, start, end) -> float:
        xy = pts[:, :2]
        tree2d = cKDTree(xy)
        sub = xy[np.linspace(0, xy.shape[0] - 1, min(2000, xy.shape[0])).astype(int)]
        d2, _ = kdtree_query(tree2d, sub, k=2, n_jobs=self.n_jobs)
        s = float(np.median(d2[:, 1]))
        if s <= 0:
            return 0.0
        a = np.asarray(start, float)[:2]
        b = np.asarray(end, float)[:2]
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            return 0.0
        u = (b - a) / L
        lat = np.array([-u[1], u[0]])
        step = s / 2.0
        t = np.arange(0.0, L + 1e-9, step)
        worst = 0
        for off in self.gap_track_offsets_m:
            line = a[None, :] + t[:, None] * u[None, :] + off * lat[None, :]
            d, _ = kdtree_query(tree2d, line, k=1, n_jobs=self.n_jobs)
            flag = np.asarray(d) > 1.5 * s
            if not flag.any():
                continue
            # longest run of consecutive True
            edges = np.diff(np.r_[0, flag.astype(int), 0])
            runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            worst = max(worst, int(runs.max()))
        # flagged run = W - 2s (minus up to one sampling step); add both back.
        # Rounding up, not down: under-reading a gap is the unsafe direction.
        return 0.0 if worst == 0 else worst * step + 2.0 * s + step

    def extract(self, segment_id, pts, start, end) -> SegmentFeatures:
        if pts.shape[0] < max(self.k + 1, 8):
            return SegmentFeatures(segment_id, start, end, 90., 90., 90., 90.,
                                   999., 999., 0., 0., 1.0, n_points=int(pts.shape[0]))
        n = min(self.n_sample, pts.shape[0])
        sidx = np.linspace(0, pts.shape[0] - 1, n).astype(int)
        normals, curv = self._normals_and_curvature(pts, sidx)
        planar = curv <= self.planar_curvature_max
        planar_fraction = float(planar.mean())
        if planar_fraction < self.min_planar_fraction:
            planar = np.ones_like(planar)          # too little planar evidence
        pn = normals[planar]
        slope = np.degrees(np.arccos(np.clip(pn[:, 2], -1, 1)))
        direction = np.asarray(end, float) - np.asarray(start, float)
        u = direction / (np.linalg.norm(direction) + 1e-9)
        rise = -(pn[:, 0] * u[0] + pn[:, 1] * u[1])
        inc, dec = slope[rise >= 0], slope[rise < 0]
        roughness = float(np.clip(curv.mean() * self.roughness_scale, 0, 1))
        along = (pts - start) @ u
        order = np.argsort(along)
        zs, a_s = pts[order, 2], along[order]
        step_legacy = float(np.max(np.abs(np.diff(zs)))) if zs.size > 1 else 0.0
        step = self._local_step_height(pts, None)
        sp = np.diff(a_s) if a_s.size > 1 else np.array([0.0])
        lat_ax = np.array([-u[1], u[0], 0.0]); lat_ax /= np.linalg.norm(lat_ax) + 1e-9
        lat = pts @ lat_ax
        width = float(np.percentile(lat, 95) - np.percentile(lat, 5))
        ground = float(np.percentile(pts[:, 2], 15))
        over = pts[pts[:, 2] > ground + 0.25]
        clear = float(np.percentile(over[:, 2], 5) - ground) if over.shape[0] else 999.0
        return SegmentFeatures(segment_id, start, end,
                               float(inc.mean()) if inc.size else 0.0,
                               float(inc.max()) if inc.size else 0.0,
                               float(dec.mean()) if dec.size else 0.0,
                               float(dec.max()) if dec.size else 0.0,
                               step,
                               (self._hole_gap(pts, start, end) if self.gap_mode == "holes"
                                else float(sp.max())),
                               clear, width, roughness,
                               n_points=int(pts.shape[0]),
                               along_track_spacing_m=float(np.median(sp)),
                               planar_fraction=planar_fraction,
                               step_legacy_m=step_legacy)


def turning_radius_from_three_points(p0, p1, p2) -> float:
    a = float(np.linalg.norm(p1[:2] - p2[:2])); b = float(np.linalg.norm(p0[:2] - p2[:2]))
    c = float(np.linalg.norm(p0[:2] - p1[:2]))
    area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
    return float("inf") if area2 < 1e-9 else float((a * b * c) / (2.0 * area2))
