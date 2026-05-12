
import csv
from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Experiment controls
# ============================================================
# Main grid
N_values = [128, 256, 512,1024,2048, 4096]
chi_values = [0.03, 0.05,0.07]
num_reps = 3

# Macroscopic horizon
T_mac = 1.0

# Initial profile rho_0(x/N): use a width proportional to N so the
# macroscopic shape is comparable across N.
rhomax = 1.0
width_fraction = 0.18

# Time mode:
#   "jump_chain"      : uses exactly floor/ceil(T_mac * N^2 * chi) jumps and dt = T_mac / num_jumps
#   "continuous_time" : simulates the continuous-time chain and integrates exactly in rescaled time
time_mode = "jump_chain"

# Plot controls
use_loglog_plot = False
output_csv = "zrp_integrated_dissipation_vs_N.csv"
output_png = "zrp_integrated_dissipation_vs_N.png"

# Reproducibility
seed = 0
rng = np.random.default_rng(seed)


# ============================================================
# Microscopic simulator (efficient raw dissipation only)
# ============================================================
def gaussian_bump(num_sites, peak_height=rhomax, width_fraction=width_fraction):
    center = num_sites // 2
    width = max(1.0, width_fraction * num_sites)
    x = np.arange(num_sites, dtype=float)
    return peak_height * np.exp(-((x - center) ** 2) / (2.0 * width ** 2))


def svle_sample(rho_profile, chi, rng):
    return rng.poisson(lam=rho_profile / chi).astype(np.int64)


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

    def total(self):
        s = 0
        i = self.n
        while i > 0:
            s += int(self.tree[i])
            i -= i & -i
        return s

    def find_by_cumulative(self, target):
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
class RawDissipationState:
    num_sites: int
    chi: float
    particles: np.ndarray
    sqrt_particles: np.ndarray
    bond_contrib: np.ndarray
    total_diss: float

    @classmethod
    def from_particles(cls, particles, chi):
        particles = np.asarray(particles, dtype=np.int64)
        num_sites = int(len(particles))
        sqrt_particles = np.sqrt(particles.astype(float))
        prefactor = num_sites * chi
        bond_contrib = prefactor * (sqrt_particles - np.roll(sqrt_particles, -1)) ** 2
        total_diss = float(np.sum(bond_contrib))
        return cls(num_sites, chi, particles, sqrt_particles, bond_contrib, total_diss)

    def update_after_jump(self, from_site, to_site):
        affected = sorted({(from_site - 1) % self.num_sites, from_site,
                           (to_site - 1) % self.num_sites, to_site})
        for bond_idx in affected:
            i = bond_idx
            j = (bond_idx + 1) % self.num_sites
            old = self.bond_contrib[bond_idx]
            new = self.num_sites * self.chi * (self.sqrt_particles[i] - self.sqrt_particles[j]) ** 2
            self.bond_contrib[bond_idx] = new
            self.total_diss += new - old


class EfficientRawZRPSimulator:
    def __init__(self, num_sites, chi, rho_profile, rng):
        self.num_sites = int(num_sites)
        self.chi = float(chi)
        self.rng = rng
        self.particles = svle_sample(rho_profile, chi, rng)
        self.tree = FenwickTree(self.particles)
        self.raw = RawDissipationState.from_particles(self.particles, chi)

    def total_rate(self):
        # For g(k)=k this equals the total number of particles and is conserved.
        return self.tree.total()

    def jump(self):
        total_particles = self.tree.total()
        if total_particles <= 0:
            return None, None

        target = int(self.rng.integers(1, total_particles + 1))
        from_site = self.tree.find_by_cumulative(target)
        to_site = int((from_site + self.rng.choice((-1, 1))) % self.num_sites)

        self.particles[from_site] -= 1
        self.particles[to_site] += 1

        self.tree.add(from_site, -1)
        self.tree.add(to_site, +1)

        self.raw.sqrt_particles[from_site] = math.sqrt(float(self.particles[from_site]))
        self.raw.sqrt_particles[to_site] = math.sqrt(float(self.particles[to_site]))
        self.raw.update_after_jump(from_site, to_site)

        return from_site, to_site


# ============================================================
# Experiment runners
# ============================================================
def simulate_integrated_dissipation(num_sites, chi, T_mac, rng, time_mode="jump_chain"):
    rho_profile = gaussian_bump(num_sites)
    sim = EfficientRawZRPSimulator(num_sites, chi, rho_profile, rng)

    if time_mode == "jump_chain":
        num_jumps = max(1, int(np.ceil(T_mac * (num_sites ** 2) * chi)))
        dt_mac = T_mac / num_jumps
        cumulative = 0.0
        for _ in range(num_jumps):
            sim.jump()
            cumulative += sim.raw.total_diss * dt_mac
        return cumulative

    if time_mode == "continuous_time":
        cumulative = 0.0
        t_mac = 0.0
        while t_mac < T_mac:
            rate = sim.total_rate()
            if rate <= 0:
                break

            # Hold current state for a rescaled exponential waiting time.
            tau = rng.exponential(scale=1.0 / rate) / (num_sites ** 2)
            dt = min(tau, T_mac - t_mac)
            cumulative += sim.raw.total_diss * dt
            t_mac += dt

            if t_mac >= T_mac:
                break

            sim.jump()
        return cumulative

    raise ValueError("time_mode must be 'jump_chain' or 'continuous_time'")


def run_grid(N_values, chi_values, num_reps, T_mac, time_mode, seed):
    results = []
    for chi_index, chi in enumerate(chi_values):
        for N in N_values:
            vals = []
            for rep in range(num_reps):
                local_seed = seed + 100000 * chi_index + 1000 * rep + int(N)
                local_rng = np.random.default_rng(local_seed)
                val = simulate_integrated_dissipation(N, chi, T_mac, local_rng, time_mode=time_mode)
                vals.append(float(val))
                print(f"done chi={chi}, N={N}, rep={rep + 1}/{num_reps}: integral={val:.6g}")
            vals = np.array(vals, dtype=float)
            results.append({
                "N": int(N),
                "chi": float(chi),
                "mean_integral": float(np.mean(vals)),
                "std_integral": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "all_values": vals,
            })
    return results


def save_csv(results, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "chi", "mean_integral", "std_integral", "replicates"])
        for row in results:
            reps = ";".join(f"{x:.10g}" for x in row["all_values"])
            writer.writerow([row["N"], row["chi"], row["mean_integral"], row["std_integral"], reps])


def plot_results(results, filename, use_loglog=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    chi_values_sorted = sorted({row["chi"] for row in results})
    for chi in chi_values_sorted:
        rows = sorted([row for row in results if row["chi"] == chi], key=lambda r: r["N"])
        N = np.array([row["N"] for row in rows], dtype=float)
        means = np.array([row["mean_integral"] for row in rows], dtype=float)
        stds = np.array([row["std_integral"] for row in rows], dtype=float)

        ax.errorbar(
            N, means, yerr=stds, marker="o", linewidth=1.2, capsize=3,
            label=rf"$\chi={chi}$"
        )

    if use_loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\int_0^1 D_N(\eta_s^N)\,ds$")
    ax.set_title("Integrated raw entropy dissipation versus system size")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main():
    results = run_grid(
        N_values=N_values,
        chi_values=chi_values,
        num_reps=num_reps,
        T_mac=T_mac,
        time_mode=time_mode,
        seed=seed,
    )
    save_csv(results, output_csv)
    plot_results(results, output_png, use_loglog=use_loglog_plot)
    print(f"Saved CSV to {output_csv}")
    print(f"Saved figure to {output_png}")


if __name__ == "__main__":
    main()
