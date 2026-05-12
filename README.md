# Multiscale-ZRP-Simulation
This repository contains code for a presentation on 'The Porous Medium Equation: Multiscale Integrability in Large Deviations' [https://arxiv.org/pdf/2602.09547], jointly with Prof. Benjamin Gess.


## Contents

├── zrp_multiscale_videos.py
├── zrp_integrated_dissipation.py
├── README.md
└── videos/
    ├── 01.mp4
    ├── 02.mp4
    ├── 03.mp4
    ├── 04.mp4
    ├── 05.mp4
    └── 06.mp4

# Requirements

The scripts require Python 3, NumPy, Matplotlib, FFmpeg.

# Multiscale ZRP Videos

This repository contains the Python code and rendered videos used in a presentation on multiscale visualizations for a zero-range-process. For ease of simulation, and because it does not change the phenomena we display, we set $\alpha=1$, and look only at the embedded jump chain rather than the full continuous-time process.

The main script generates six synchronised animations showing the evolution of a zero-range process together with coarse-grained observables and entropy-dissipation diagnostics across several spatial scales. The six animations generated are as follows:
    01.mp4: the underlying particle configuration with no local averaging;
    02.mp4: local averaging, on a scale still too small to gain regularity;
    03.mp4: local averaging, on a scale large enough to gain regularity;
    04.mp4: within each block of local averaging, the particle configuration remains irregular;
    05.mp4: further local averaging within each block;
    06.mp4: a complete hierarchy of scales, within each of which the local averages of the next scale are well-behaved.

The initial parameters are set to

num_sites = 10000
chi = 0.5
rhomax = 1.0
initial_width = 200

ell_bad = 5
ell_good = 100

T_mac = 1

The code is set by default to simulate up to a macroscopic time, given by 

num_jumps = int(np.ceil(T_mac * (num_sites ** 2) / chi))

For a diagnostic, comment this line out, and uncomment the line

num_jumps = 10

The initial profile is a Gaussian-type bump centered in the domain. A particle configuration is sampled from this profile, then evolved by repeated nearest-neighbor jumps on a periodic lattice. In order to reduce computational load, only the local averages, and the associated contributions to entropy dissipation, which are changed by the jump are updated.


# ZRP Integrated Dissipation

This file produces a plot of the integrated discrete entropy dissipation, when no local averaging is performed, for different values of N, chi, and a fixed macroscopic time T_mac. 
