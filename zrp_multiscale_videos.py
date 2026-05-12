import contextlib
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle, ConnectionPatch

# ============================================================
# Parameters
# ============================================================
num_sites = 10000
chi = 0.05
rhomax = 1.0
initial_width = 200

ell_bad = 10
ell_good = 400

T_mac = 1
# Seminar-display defaults.
# To recover an "exact" display mode close to v10, set:
#   display_smoothing_alpha = 1.0
#   display_integrated_entropy_dissipation = True
#   num_frames = 180
#   fps = 20
#   hold_initial_frames = 1
#   hold_final_frames = 1
num_frames = 80
fps = 6

# Display controls
display_integrated_entropy_dissipation = False
display_smoothing_alpha = 0.18   # 1.0 means no smoothing of displayed instantaneous values
hold_initial_frames = 6
hold_final_frames = 10

# Fixed zoom choices.
# These are block indices, not site indices.
# The defaults pick the central good block and the central bad block inside it.
fixed_good_block = (num_sites // ell_good) // 2
fixed_small_block_in_good = (ell_good // ell_bad) // 2

# For the jump-chain surrogate used here, macroscopic time 1 corresponds to
# order N^2 / chi jumps.
num_jumps = int(np.ceil(T_mac * (num_sites ** 2) / chi))

# Use for a smoke test
# num_jumps = 10

seed = 0
rng = np.random.default_rng(seed)


# ============================================================
# Styling
# ============================================================
GREY = "0.60"
GREY_LIGHT = "0.82"
RED = "firebrick"
BLACK = "black"

LW_BG = 0.70
LW_FG = 1.35
LW_FINE = 0.85

# ============================================================
# Helpers
# ============================================================
def gaussian_bump(num_sites, peak_height=rhomax, width=initial_width):
    center = num_sites // 2
    x = np.arange(num_sites)
    return peak_height * (1 + 4* np.exp(-((x - center) ** 2) / (2 * width ** 2))) / 5


def svle_sample(rho_profile, chi, rng):
    return rng.poisson(lam=rho_profile / chi).astype(np.int64)


def fmt(x):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exponent)
    if exponent == 0:
        return rf"{mantissa:.2f}"
    return rf"{mantissa:.2f}\times 10^{{{exponent}}}"


def minimal_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")


def add_stat_lines(fig, ax, lines, colors, fontsize=9, line_step=0.045,
                   inside_loc=(0.02, 0.97), outside_gap=0.014, min_left_space=0.14):
    old_inside = getattr(ax, "_stat_text_artists", [])
    for artist in old_inside:
        try:
            artist.remove()
        except ValueError:
            pass
    ax._stat_text_artists = []

    old_outside = getattr(ax, "_outside_stat_text_artists", [])
    for artist in old_outside:
        try:
            artist.remove()
        except ValueError:
            pass
    ax._outside_stat_text_artists = []

    pos = ax.get_position()
    if pos.x0 >= min_left_space:
        x = pos.x0 - outside_gap
        y = pos.y1
        for idx, (line, color) in enumerate(zip(lines, colors)):
            artist = fig.text(
                x,
                y - line_step * idx,
                line,
                va="top",
                ha="right",
                color=color,
                fontsize=fontsize,
            )
            ax._outside_stat_text_artists.append(artist)
    else:
        x0, y0 = inside_loc
        for idx, (line, color) in enumerate(zip(lines, colors)):
            bbox = None
            if idx == 0:
                bbox = dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="none")
            artist = ax.text(
                x0,
                y0 - 1.75 * line_step * idx,
                line,
                transform=ax.transAxes,
                va="top",
                ha="left",
                color=color,
                fontsize=fontsize,
                bbox=bbox,
            )
            ax._stat_text_artists.append(artist)


def sym_raw():
    return r"\eta_t^N"


def sym_lambda(ell):
    return rf"\Lambda_{{B_{{{ell}}}}}\eta_t^N"


def line_compact(expr, inst, cum, show_integrated=True):
    if show_integrated:
        return rf"${expr}:\ \mathcal{{D}}={fmt(inst)},\ \int_0^t \mathcal{{D}}\,ds={fmt(cum)}$"
    return rf"${expr}:\ \mathcal{{D}}={fmt(inst)}$"


def update_display_value(previous, current, alpha):
    if alpha is None or alpha >= 1.0:
        return float(current)
    if alpha <= 0.0:
        return float(previous)
    return float((1.0 - alpha) * previous + alpha * current)


def expr_avg_block_raw(ell_outer):
    return rf"\operatorname{{Av}}_{{B\in\mathcal{{P}}_{{{ell_outer}}}}}\mathcal{{D}}_B(\eta_t^N)"


def expr_avg_block_lambda(ell_outer, ell_inner):
    return rf"\operatorname{{Av}}_{{B\in\mathcal{{P}}_{{{ell_outer}}}}}\mathcal{{D}}_B(\Lambda_{{B_{{{ell_inner}}}}}\eta_t^N)"


def expr_avg_subblocks(ell_inner):
    return rf"\operatorname{{Av}}_{{B\in\mathcal{{P}}_{{{ell_inner}}}(B^\sharp)}}\mathcal{{D}}_B(\eta_t^N)"


def make_zoom_figure_two():
    fig = plt.figure(figsize=(12, 6.0))
    ax_top = fig.add_axes([0.20, 0.50, 0.74, 0.38])
    ax_bottom = fig.add_axes([0.50, 0.11, 0.18, 0.16])
    return fig, ax_top, ax_bottom


def make_zoom_figure_three():
    fig = plt.figure(figsize=(12, 7.3))
    ax_top = fig.add_axes([0.20, 0.63, 0.74, 0.25])
    ax_mid = fig.add_axes([0.50, 0.34, 0.18, 0.15])
    ax_bottom = fig.add_axes([0.545, 0.08, 0.09, 0.11])
    return fig, ax_top, ax_mid, ax_bottom


def add_zoom_rectangle(ax, start, end, ymax, edgecolor="0.15", linewidth=1.2):
    rect = Rectangle((start, 0.0), end - start, ymax, fill=False, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(rect)


def add_connector(fig, ax_from, xy_from, ax_to, xy_to, color="0.35", linewidth=0.9):
    conn = ConnectionPatch(
        xyA=xy_from,
        coordsA=ax_from.transData if ax_from is not None else None,
        xyB=xy_to,
        coordsB=ax_to.transAxes,
        color=color,
        linewidth=linewidth,
        alpha=0.9,
    )
    fig.add_artist(conn)


# ============================================================
# Efficient data structures
# ============================================================
class FenwickTree:
    def __init__(self, values):
        values = np.asarray(values, dtype=np.int64)
        self.n = int(len(values))
        self.tree = np.zeros(self.n + 1, dtype=np.int64)
        for idx, val in enumerate(values):
            self.add(idx, int(val))

    def add(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, index):
        s = 0
        i = index + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return int(s)

    def total(self):
        return int(self.prefix_sum(self.n - 1))

    def find_by_cumulative(self, target):
        # target in {1, ..., total}
        idx = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= self.n and self.tree[nxt] < target:
                target -= int(self.tree[nxt])
                idx = nxt
            bit >>= 1
        return idx


@dataclass
class EqualBlockState:
    num_sites: int
    ell: int
    chi: float
    block_values: np.ndarray
    expanded_field: np.ndarray
    bond_contrib: np.ndarray
    total_diss: float

    def __post_init__(self):
        if self.num_sites % self.ell != 0:
            raise ValueError(f"ell={self.ell} must divide num_sites={self.num_sites} in this efficient prototype.")
        self.num_blocks = self.num_sites // self.ell
        self.starts = np.arange(0, self.num_sites, self.ell, dtype=np.int64)
        self.ends = self.starts + self.ell

    @classmethod
    def from_particles(cls, particles, ell, chi):
        particles = np.asarray(particles, dtype=np.int64)
        num_sites = int(len(particles))
        if num_sites % ell != 0:
            raise ValueError(f"ell={ell} must divide num_sites={num_sites}.")
        num_blocks = num_sites // ell
        block_mass = particles.reshape(num_blocks, ell).sum(axis=1).astype(float)
        block_values = chi * block_mass / ell
        expanded_field = np.repeat(block_values, ell)
        sqrt_vals = np.sqrt(np.clip(block_values, 0.0, None))
        bond_contrib = num_blocks * (sqrt_vals - np.roll(sqrt_vals, -1)) ** 2
        total_diss = float(np.sum(bond_contrib))
        return cls(num_sites, ell, chi, block_values, expanded_field, bond_contrib, total_diss)

    def site_to_block(self, site):
        return site // self.ell

    def update_for_jump(self, from_site, to_site, nested_outer=None):
        from_block = self.site_to_block(from_site)
        to_block = self.site_to_block(to_site)
        if from_block == to_block:
            return from_block, to_block, []

        delta_val = self.chi / self.ell

        self.block_values[from_block] -= delta_val
        self.block_values[to_block] += delta_val

        fs = from_block * self.ell
        ts = to_block * self.ell
        self.expanded_field[fs:fs + self.ell] = self.block_values[from_block]
        self.expanded_field[ts:ts + self.ell] = self.block_values[to_block]

        affected = sorted({(from_block - 1) % self.num_blocks, from_block,
                           (to_block - 1) % self.num_blocks, to_block})
        for bond_idx in affected:
            old = self.bond_contrib[bond_idx]
            left = self.block_values[bond_idx]
            right = self.block_values[(bond_idx + 1) % self.num_blocks]
            new = self.num_blocks * (np.sqrt(max(left, 0.0)) - np.sqrt(max(right, 0.0))) ** 2
            delta = new - old
            self.bond_contrib[bond_idx] = new
            self.total_diss += delta
            if nested_outer is not None and nested_outer.inner_bond_is_internal[bond_idx]:
                nested_outer.inner_internal_sum[nested_outer.inner_bond_to_outer_block[bond_idx]] += delta

        return from_block, to_block, affected


@dataclass
class NestedOuterDiagnostics:
    # Stores, for each outer block, the sum of inner coarse-bond contributions
    # across those inner bonds that lie strictly inside the outer block.
    inner_internal_sum: np.ndarray
    inner_bond_is_internal: np.ndarray
    inner_bond_to_outer_block: np.ndarray
    scale_factor: float

    @classmethod
    def from_states(cls, inner_state, outer_state):
        if outer_state.ell % inner_state.ell != 0:
            raise ValueError("Need inner ell to divide outer ell.")
        ratio = outer_state.ell // inner_state.ell
        inner_bond_is_internal = np.zeros(inner_state.num_blocks, dtype=bool)
        inner_bond_to_outer_block = -np.ones(inner_state.num_blocks, dtype=np.int64)
        inner_internal_sum = np.zeros(outer_state.num_blocks, dtype=float)

        for j in range(inner_state.num_blocks):
            nxt = (j + 1) % inner_state.num_blocks
            left_outer = j // ratio
            right_outer = nxt // ratio if nxt != 0 else 0
            is_internal = (j != inner_state.num_blocks - 1) and (left_outer == right_outer)
            if is_internal:
                inner_bond_is_internal[j] = True
                inner_bond_to_outer_block[j] = left_outer
                inner_internal_sum[left_outer] += inner_state.bond_contrib[j]

        scale_factor = ratio / inner_state.num_blocks
        return cls(inner_internal_sum, inner_bond_is_internal, inner_bond_to_outer_block, scale_factor)


@dataclass
class RawDissipationState:
    num_sites: int
    chi: float
    particles: np.ndarray
    sqrt_particles: np.ndarray
    bond_contrib: np.ndarray
    total_diss: float
    ell_bad: int
    ell_good: int
    bad_internal_sum: np.ndarray
    good_internal_sum: np.ndarray

    @classmethod
    def from_particles(cls, particles, chi, ell_bad, ell_good):
        particles = np.asarray(particles, dtype=np.int64)
        num_sites = int(len(particles))
        sqrt_particles = np.sqrt(particles.astype(float))
        bond_contrib = np.empty(num_sites, dtype=float)
        prefactor = num_sites * chi
        for i in range(num_sites):
            j = (i + 1) % num_sites
            bond_contrib[i] = prefactor * (sqrt_particles[i] - sqrt_particles[j]) ** 2
        total_diss = float(np.sum(bond_contrib))

        if num_sites % ell_bad != 0 or num_sites % ell_good != 0:
            raise ValueError("ell_bad and ell_good must divide num_sites.")

        bad_internal_sum = np.zeros(num_sites // ell_bad, dtype=float)
        for b in range(num_sites // ell_bad):
            start = b * ell_bad
            bad_internal_sum[b] = np.sum(bond_contrib[start:start + ell_bad - 1])

        good_internal_sum = np.zeros(num_sites // ell_good, dtype=float)
        for b in range(num_sites // ell_good):
            start = b * ell_good
            good_internal_sum[b] = np.sum(bond_contrib[start:start + ell_good - 1])

        return cls(num_sites, chi, particles, sqrt_particles, bond_contrib, total_diss,
                   ell_bad, ell_good, bad_internal_sum, good_internal_sum)

    def _bond_internal(self, bond_idx, ell):
        return (bond_idx != self.num_sites - 1) and (((bond_idx + 1) % ell) != 0)

    def update_after_jump(self, from_site, to_site):
        affected = sorted({(from_site - 1) % self.num_sites, from_site,
                           (to_site - 1) % self.num_sites, to_site})
        for bond_idx in affected:
            old = self.bond_contrib[bond_idx]
            i = bond_idx
            j = (bond_idx + 1) % self.num_sites
            new = self.num_sites * self.chi * (self.sqrt_particles[i] - self.sqrt_particles[j]) ** 2
            delta = new - old
            self.bond_contrib[bond_idx] = new
            self.total_diss += delta
            if self._bond_internal(bond_idx, self.ell_bad):
                self.bad_internal_sum[bond_idx // self.ell_bad] += delta
            if self._bond_internal(bond_idx, self.ell_good):
                self.good_internal_sum[bond_idx // self.ell_good] += delta


class EfficientZRPSimulator:
    def __init__(self, num_sites, chi, ell_bad, ell_good, rho_profile, rng):
        self.num_sites = num_sites
        self.chi = chi
        self.ell_bad = ell_bad
        self.ell_good = ell_good
        self.rng = rng

        self.particles = svle_sample(rho_profile, chi, rng)
        self.tree = FenwickTree(self.particles)
        self.raw = RawDissipationState.from_particles(self.particles, chi, ell_bad, ell_good)
        self.bad_state = EqualBlockState.from_particles(self.particles, ell_bad, chi)
        self.good_state = EqualBlockState.from_particles(self.particles, ell_good, chi)
        self.nested_good_of_bad = NestedOuterDiagnostics.from_states(self.bad_state, self.good_state)

    def rho(self):
        return self.chi * self.particles.astype(float)

    def jump(self):
        total_particles = self.tree.total()
        if total_particles == 0:
            return None, None

        target = int(self.rng.integers(1, total_particles + 1))
        from_site = self.tree.find_by_cumulative(target)
        to_site = int((from_site + self.rng.choice((-1, 1))) % self.num_sites)

        # Microscopic update
        self.particles[from_site] -= 1
        self.particles[to_site] += 1
        self.tree.add(from_site, -1)
        self.tree.add(to_site, +1)
        self.raw.sqrt_particles[from_site] = np.sqrt(float(self.particles[from_site]))
        self.raw.sqrt_particles[to_site] = np.sqrt(float(self.particles[to_site]))

        # Coarse updates
        self.bad_state.update_for_jump(from_site, to_site, nested_outer=self.nested_good_of_bad)
        self.good_state.update_for_jump(from_site, to_site, nested_outer=None)

        # Raw dissipation update
        self.raw.update_after_jump(from_site, to_site)

        return from_site, to_site


# ============================================================
# Drawing routines
# ============================================================
def draw_video_1(fig, ax, rho, raw_inst, raw_cum, ymax):
    ax.clear()
    ax.plot(rho, color=RED, linewidth=LW_FINE)
    minimal_axis(ax, (0, len(rho) - 1), (0.0, ymax))
    add_stat_lines(fig, ax, [line_compact(sym_raw(), raw_inst, raw_cum, display_integrated_entropy_dissipation)], [RED], fontsize=9.2, line_step=0.050)



def draw_video_2(fig, ax, rho, bad_field, raw_inst, raw_cum, bad_inst, bad_cum, ymax):
    ax.clear()
    ax.plot(rho, color=GREY, linewidth=LW_BG)
    ax.plot(bad_field, color=RED, linewidth=LW_FG)
    minimal_axis(ax, (0, len(rho) - 1), (0.0, ymax))
    add_stat_lines(
        fig, ax,
        [line_compact(sym_raw(), raw_inst, raw_cum, display_integrated_entropy_dissipation), line_compact(sym_lambda(ell_bad), bad_inst, bad_cum, display_integrated_entropy_dissipation)],
        [GREY, RED], fontsize=8.4, line_step=0.054,
    )



def draw_video_3(fig, ax, rho, good_field, raw_inst, raw_cum, good_inst, good_cum, ymax):
    ax.clear()
    ax.plot(rho, color=GREY, linewidth=LW_BG)
    ax.plot(good_field, color=BLACK, linewidth=LW_FG)
    minimal_axis(ax, (0, len(rho) - 1), (0.0, ymax))
    add_stat_lines(
        fig, ax,
        [line_compact(sym_raw(), raw_inst, raw_cum, display_integrated_entropy_dissipation), line_compact(sym_lambda(ell_good), good_inst, good_cum, display_integrated_entropy_dissipation)],
        [GREY, BLACK], fontsize=8.4, line_step=0.054,
    )



def draw_video_4(fig, ax_top, ax_bottom, rho, good_field, raw_inst, raw_cum, good_inst, good_cum,
                 local_raw, active_start, active_end, intrablock_raw_inst, intrablock_raw_cum,
                 ymax_top, ymax_bottom):
    ax_top.clear()
    ax_bottom.clear()

    ax_top.plot(rho, color=GREY, linewidth=LW_BG)
    ax_top.plot(good_field, color=BLACK, linewidth=LW_FG)
    minimal_axis(ax_top, (0, len(rho) - 1), (0.0, ymax_top))
    add_stat_lines(
        fig, ax_top,
        [line_compact(sym_raw(), raw_inst, raw_cum, display_integrated_entropy_dissipation), line_compact(sym_lambda(ell_good), good_inst, good_cum, display_integrated_entropy_dissipation)],
        [GREY, BLACK], fontsize=8.0, line_step=0.050,
    )

    ax_bottom.plot(local_raw, color=RED, linewidth=LW_FINE)
    minimal_axis(ax_bottom, (0, len(local_raw) - 1), (0.0, ymax_bottom))
    add_stat_lines(
        fig, ax_bottom,
        [line_compact(expr_avg_block_raw(ell_good), intrablock_raw_inst, intrablock_raw_cum, display_integrated_entropy_dissipation)],
        [RED], fontsize=7.8, line_step=0.048,
    )

    add_zoom_rectangle(ax_top, active_start, active_end, ymax_top)
    add_connector(fig, ax_top, (active_start, 0.0), ax_bottom, (0.0, 1.0))
    add_connector(fig, ax_top, (active_end, 0.0), ax_bottom, (1.0, 1.0))



def draw_video_5(fig, ax_top, ax_bottom, rho, good_field, raw_inst, raw_cum, good_inst, good_cum,
                 local_raw, local_small, local_small_boundaries, active_start, active_end,
                 intrablock_raw_inst, intrablock_raw_cum, intrablock_small_inst, intrablock_small_cum,
                 ymax_top, ymax_bottom):
    ax_top.clear()
    ax_bottom.clear()

    ax_top.plot(rho, color=GREY, linewidth=LW_BG)
    ax_top.plot(good_field, color=BLACK, linewidth=LW_FG)
    minimal_axis(ax_top, (0, len(rho) - 1), (0.0, ymax_top))
    add_stat_lines(
        fig, ax_top,
        [line_compact(sym_raw(), raw_inst, raw_cum, display_integrated_entropy_dissipation), line_compact(sym_lambda(ell_good), good_inst, good_cum, display_integrated_entropy_dissipation)],
        [GREY, BLACK], fontsize=8.0, line_step=0.050,
    )

    ax_bottom.plot(local_raw, color=GREY, linewidth=LW_BG)
    ax_bottom.plot(local_small, color=BLACK, linewidth=LW_FG)
    for boundary in local_small_boundaries:
        ax_bottom.axvline(boundary - 0.5, color=GREY_LIGHT, linestyle="--", linewidth=0.45, alpha=0.8)
    minimal_axis(ax_bottom, (0, len(local_raw) - 1), (0.0, ymax_bottom))
    add_stat_lines(
        fig, ax_bottom,
        [line_compact(expr_avg_block_raw(ell_good), intrablock_raw_inst, intrablock_raw_cum, display_integrated_entropy_dissipation),
         line_compact(expr_avg_block_lambda(ell_good, ell_bad), intrablock_small_inst, intrablock_small_cum, display_integrated_entropy_dissipation)],
        [GREY, BLACK], fontsize=7.2, line_step=0.046,
    )

    add_zoom_rectangle(ax_top, active_start, active_end, ymax_top)
    add_connector(fig, ax_top, (active_start, 0.0), ax_bottom, (0.0, 1.0))
    add_connector(fig, ax_top, (active_end, 0.0), ax_bottom, (1.0, 1.0))



def draw_video_6(fig, ax_top, ax_mid, ax_bottom, rho, good_field, raw_inst, raw_cum, good_inst, good_cum,
                 local_raw, local_small, local_small_boundaries, local_tiny_raw,
                 active_good_start, active_good_end, active_small_start, active_small_end,
                 intrablock_raw_inst, intrablock_raw_cum, intrablock_small_inst, intrablock_small_cum,
                 tiny_raw_avg_inst, tiny_raw_avg_cum,
                 ymax_top, ymax_mid, ymax_bottom):
    ax_top.clear()
    ax_mid.clear()
    ax_bottom.clear()

    ax_top.plot(rho, color=GREY, linewidth=LW_BG)
    ax_top.plot(good_field, color=BLACK, linewidth=LW_FG)
    minimal_axis(ax_top, (0, len(rho) - 1), (0.0, ymax_top))
    add_stat_lines(
        fig, ax_top,
        [line_compact(sym_raw(), raw_inst, raw_cum, display_integrated_entropy_dissipation), line_compact(sym_lambda(ell_good), good_inst, good_cum, display_integrated_entropy_dissipation)],
        [GREY, BLACK], fontsize=7.8, line_step=0.048,
    )

    ax_mid.plot(local_raw, color=GREY, linewidth=LW_BG)
    ax_mid.plot(local_small, color=BLACK, linewidth=LW_FG)
    for boundary in local_small_boundaries:
        ax_mid.axvline(boundary - 0.5, color=GREY_LIGHT, linestyle="--", linewidth=0.45, alpha=0.8)
    minimal_axis(ax_mid, (0, len(local_raw) - 1), (0.0, ymax_mid))
    add_stat_lines(
        fig, ax_mid,
        [line_compact(expr_avg_block_raw(ell_good), intrablock_raw_inst, intrablock_raw_cum, display_integrated_entropy_dissipation),
         line_compact(expr_avg_block_lambda(ell_good, ell_bad), intrablock_small_inst, intrablock_small_cum, display_integrated_entropy_dissipation)],
        [GREY, BLACK], fontsize=6.9, line_step=0.044,
    )

    ax_bottom.plot(local_tiny_raw, color=BLACK, linewidth=1.15)
    minimal_axis(ax_bottom, (0, len(local_tiny_raw) - 1), (0.0, ymax_bottom))
    add_stat_lines(
        fig, ax_bottom,
        [line_compact(expr_avg_subblocks(ell_bad), tiny_raw_avg_inst, tiny_raw_avg_cum, display_integrated_entropy_dissipation)],
        [BLACK], fontsize=6.8, line_step=0.044,
    )

    add_zoom_rectangle(ax_top, active_good_start, active_good_end, ymax_top)
    add_connector(fig, ax_top, (active_good_start, 0.0), ax_mid, (0.0, 1.0))
    add_connector(fig, ax_top, (active_good_end, 0.0), ax_mid, (1.0, 1.0))

    local_small_start = active_small_start - active_good_start
    local_small_end = active_small_end - active_good_start
    add_zoom_rectangle(ax_mid, local_small_start, local_small_end, ymax_mid, edgecolor="0.20", linewidth=1.1)
    add_connector(fig, ax_mid, (local_small_start, 0.0), ax_bottom, (0.0, 1.0))
    add_connector(fig, ax_mid, (local_small_end, 0.0), ax_bottom, (1.0, 1.0))


# ============================================================
# Diagnostics built from efficient state
# ============================================================
def gather_diagnostics(sim):
    rho = sim.rho()
    bad_field = sim.bad_state.expanded_field
    good_field = sim.good_state.expanded_field

    raw_inst = float(sim.raw.total_diss)
    bad_inst = float(sim.bad_state.total_diss)
    good_inst = float(sim.good_state.total_diss)

    # Average over good blocks of the raw dissipation within each good block.
    intrablock_raw_inst = float((sim.ell_good / sim.num_sites) * np.mean(sim.raw.good_internal_sum))

    # Average over good blocks of the bad-scale coarse dissipation within each good block.
    intrablock_small_inst = float(sim.nested_good_of_bad.scale_factor * np.mean(sim.nested_good_of_bad.inner_internal_sum))

    num_good_blocks = sim.num_sites // sim.ell_good
    num_small_per_good = sim.ell_good // sim.ell_bad

    if not (0 <= fixed_good_block < num_good_blocks):
        raise ValueError(f"fixed_good_block={fixed_good_block} must lie in {{0, ..., {num_good_blocks - 1}}}.")
    if not (0 <= fixed_small_block_in_good < num_small_per_good):
        raise ValueError(
            f"fixed_small_block_in_good={fixed_small_block_in_good} must lie in "
            f"{{0, ..., {num_small_per_good - 1}}}."
        )

    active_good_block = int(fixed_good_block)
    active_good_start = active_good_block * sim.ell_good
    active_good_end = active_good_start + sim.ell_good

    local_raw = rho[active_good_start:active_good_end]
    local_small = sim.bad_state.expanded_field[active_good_start:active_good_end]

    first_small = active_good_start // sim.ell_bad
    local_small_boundaries = np.arange(sim.ell_bad, sim.ell_good, sim.ell_bad)

    active_small_block = first_small + int(fixed_small_block_in_good)
    active_small_start = active_small_block * sim.ell_bad
    active_small_end = active_small_start + sim.ell_bad
    local_tiny_raw = rho[active_small_start:active_small_end]

    raw_bad_blocks_inside_good = (sim.ell_bad / sim.num_sites) * sim.raw.bad_internal_sum[first_small:first_small + num_small_per_good]
    tiny_raw_avg_inst = float(np.mean(raw_bad_blocks_inside_good))

    return {
        "rho": rho,
        "bad_field": bad_field,
        "good_field": good_field,
        "raw_inst": raw_inst,
        "bad_inst": bad_inst,
        "good_inst": good_inst,
        "intrablock_raw_inst": intrablock_raw_inst,
        "intrablock_small_inst": intrablock_small_inst,
        "active_good_start": active_good_start,
        "active_good_end": active_good_end,
        "active_small_start": active_small_start,
        "active_small_end": active_small_end,
        "local_raw": local_raw,
        "local_small": local_small,
        "local_small_boundaries": local_small_boundaries,
        "local_tiny_raw": local_tiny_raw,
        "tiny_raw_avg_inst": tiny_raw_avg_inst,
    }


# ============================================================
# Main rendering pipeline
# ============================================================
def main():
    rho_profile = gaussian_bump(num_sites)
    sim = EfficientZRPSimulator(num_sites, chi, ell_bad, ell_good, rho_profile, rng)

    dt_mac = T_mac / num_jumps if num_jumps > 0 else 0.0
    frame_times = np.linspace(0.0, T_mac, num_frames)
    next_frame = 0
    current_time = 0.0

    rho_initial = sim.rho()
    ymax_top = 1.15 * max(rhomax, float(np.max(rho_initial)))
    ymax_mid = 1.15 * float(np.max(rho_initial))
    ymax_bottom = 1.15 * float(np.max(rho_initial))

    fig1 = plt.figure(figsize=(12, 4))
    ax1 = fig1.add_axes([0.20, 0.14, 0.74, 0.76])
    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_axes([0.20, 0.14, 0.74, 0.76])
    fig3 = plt.figure(figsize=(12, 4))
    ax3 = fig3.add_axes([0.20, 0.14, 0.74, 0.76])
    fig4, ax4_top, ax4_bottom = make_zoom_figure_two()
    fig5, ax5_top, ax5_bottom = make_zoom_figure_two()
    fig6, ax6_top, ax6_mid, ax6_bottom = make_zoom_figure_three()

    metadata = dict(
        title="Multiscale ZRP",
        artist="Gess, Heydecker",
    )

    writer1 = FFMpegWriter(fps=fps, metadata=metadata)
    writer2 = FFMpegWriter(fps=fps, metadata=metadata)
    writer3 = FFMpegWriter(fps=fps, metadata=metadata)
    writer4 = FFMpegWriter(fps=fps, metadata=metadata)
    writer5 = FFMpegWriter(fps=fps, metadata=metadata)
    writer6 = FFMpegWriter(fps=fps, metadata=metadata)

    cumulative = {
        "raw": 0.0,
        "bad": 0.0,
        "good": 0.0,
        "intrablock_raw": 0.0,
        "intrablock_small": 0.0,
        "tiny_raw_avg": 0.0,
    }

    display_inst = {
        "raw": 0.0,
        "bad": 0.0,
        "good": 0.0,
        "intrablock_raw": 0.0,
        "intrablock_small": 0.0,
        "tiny_raw_avg": 0.0,
    }

    def draw_all(stats):
        draw_video_1(fig1, ax1, stats["rho"], display_inst["raw"], cumulative["raw"], ymax_top)
        writer1.grab_frame()

        draw_video_2(fig2, ax2, stats["rho"], stats["bad_field"], display_inst["raw"], cumulative["raw"],
                     display_inst["bad"], cumulative["bad"], ymax_top)
        writer2.grab_frame()

        draw_video_3(fig3, ax3, stats["rho"], stats["good_field"], display_inst["raw"], cumulative["raw"],
                     display_inst["good"], cumulative["good"], ymax_top)
        writer3.grab_frame()

        draw_video_4(fig4, ax4_top, ax4_bottom,
                     stats["rho"], stats["good_field"],
                     display_inst["raw"], cumulative["raw"],
                     display_inst["good"], cumulative["good"],
                     stats["local_raw"], stats["active_good_start"], stats["active_good_end"],
                     display_inst["intrablock_raw"], cumulative["intrablock_raw"],
                     ymax_top, ymax_mid)
        writer4.grab_frame()

        draw_video_5(fig5, ax5_top, ax5_bottom,
                     stats["rho"], stats["good_field"],
                     display_inst["raw"], cumulative["raw"],
                     display_inst["good"], cumulative["good"],
                     stats["local_raw"], stats["local_small"], stats["local_small_boundaries"],
                     stats["active_good_start"], stats["active_good_end"],
                     display_inst["intrablock_raw"], cumulative["intrablock_raw"],
                     display_inst["intrablock_small"], cumulative["intrablock_small"],
                     ymax_top, ymax_mid)
        writer5.grab_frame()

        draw_video_6(fig6, ax6_top, ax6_mid, ax6_bottom,
                     stats["rho"], stats["good_field"],
                     display_inst["raw"], cumulative["raw"],
                     display_inst["good"], cumulative["good"],
                     stats["local_raw"], stats["local_small"], stats["local_small_boundaries"],
                     stats["local_tiny_raw"],
                     stats["active_good_start"], stats["active_good_end"],
                     stats["active_small_start"], stats["active_small_end"],
                     display_inst["intrablock_raw"], cumulative["intrablock_raw"],
                     display_inst["intrablock_small"], cumulative["intrablock_small"],
                     display_inst["tiny_raw_avg"], cumulative["tiny_raw_avg"],
                     ymax_top, ymax_mid, ymax_bottom)
        writer6.grab_frame()

    with contextlib.ExitStack() as stack:
        stack.enter_context(writer1.saving(fig1, f"01.mp4", dpi=140))
        stack.enter_context(writer2.saving(fig2, f"02.mp4", dpi=140))
        stack.enter_context(writer3.saving(fig3, f"03.mp4", dpi=140))
        stack.enter_context(writer4.saving(fig4, f"04.mp4", dpi=140))
        stack.enter_context(writer5.saving(fig5, f"05.mp4", dpi=140))
        stack.enter_context(writer6.saving(fig6, f"06.mp4", dpi=140))

        stats = gather_diagnostics(sim)
        for key in display_inst:
            display_inst[key] = float(stats[f"{key}_inst"])
        for _ in range(max(1, hold_initial_frames)):
            draw_all(stats)
        next_frame = 1

        for jump in range(1, num_jumps + 1):
            sim.jump()
            stats = gather_diagnostics(sim)

            cumulative["raw"] += stats["raw_inst"] * dt_mac
            cumulative["bad"] += stats["bad_inst"] * dt_mac
            cumulative["good"] += stats["good_inst"] * dt_mac
            cumulative["intrablock_raw"] += stats["intrablock_raw_inst"] * dt_mac
            cumulative["intrablock_small"] += stats["intrablock_small_inst"] * dt_mac
            cumulative["tiny_raw_avg"] += stats["tiny_raw_avg_inst"] * dt_mac

            display_inst["raw"] = update_display_value(display_inst["raw"], stats["raw_inst"], display_smoothing_alpha)
            display_inst["bad"] = update_display_value(display_inst["bad"], stats["bad_inst"], display_smoothing_alpha)
            display_inst["good"] = update_display_value(display_inst["good"], stats["good_inst"], display_smoothing_alpha)
            display_inst["intrablock_raw"] = update_display_value(display_inst["intrablock_raw"], stats["intrablock_raw_inst"], display_smoothing_alpha)
            display_inst["intrablock_small"] = update_display_value(display_inst["intrablock_small"], stats["intrablock_small_inst"], display_smoothing_alpha)
            display_inst["tiny_raw_avg"] = update_display_value(display_inst["tiny_raw_avg"], stats["tiny_raw_avg_inst"], display_smoothing_alpha)
            current_time += dt_mac

            while next_frame < num_frames and current_time >= frame_times[next_frame]:
                draw_all(stats)
                next_frame += 1

            if jump % max(1, num_jumps // 20) == 0:
                print(f"{jump} of {num_jumps}")

        while next_frame < num_frames:
            draw_all(stats)
            next_frame += 1

        for _ in range(max(0, hold_final_frames - 1)):
            draw_all(stats)

    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        plt.close(fig)


if __name__ == "__main__":
    main()
