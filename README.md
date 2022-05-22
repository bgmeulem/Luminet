# Luminet
This repo provides material for recreating the famous paper by Jean-Pierre Luminet, simulating the first image of a black hole. This is done in Python 3.8
<img src="SampledPoints_incl=10.png" alt="Picture" />
<img src="output.gif" alt="gif" />


# Usage

```python
from black_hole import *

M = 1.
bh = BlackHole(inclination=80, mass=M)

# Calculate single isoradial
ir = Isoradial(R=30 * M, incl=80 * np.pi / 180, mass=M, order=0)
ir.calculate()
ir.plot_redshift()  # plot its redshifts along the line

# plot isoradial lines
bh.plotIsoradials([6, 10, 20, 30], [6, 10, 20, 30])

# write frames for a gif of rotating isoradial lines
bh.writeFrames(direct_r=[6, 10, 20, 30], ghost_r=[6, 10, 20, 30], start=0, end=180, step_size=5,
               ax_lim=(-35, 35))

# plot lines of equal redshift values
bh.plotIsoRedshifts(minR=5, maxR=80, r_precision=20, midpoint_steps=5,
                    redshifts=[-.5, -.35, -.15, 0., .15, .25, .5, .75, 1.])
```

# Latest updates:
Rewrite of entire branch. Apologies for the lack of incremental updates, it seems my upstream branch has not been set up properly. This is fixed now.
Everything has been moved to two files: black_hole_math.py (for calculating equations and variables) and a file containing the class 
BlackHole() for easy access to calculations.

Vastly improved speed and stability. Impact parameters are calculated using the midpoint method: stable and quick.
Added functionality to calculate redshifts and isoredshift lines

(30/12/2021)
- Identified errors in the paper. This explains the large redshift values. Fixed these errors and annotated them in the code.
- Added functionality to plot over the original figure to check for these errors (only the figure of isoradial lines at incl=60°)

(24/2/2022)
- Fixed redshift
- Can now sample points in (R, alpha) space. Luminet started from the isofluxlines though, which may be (will probably be) more efficient.

(20/5/2022) 
- changed from mpmath library to scipy for the calculation of elliptic integrals. Mpmath was only useful for support of complex solutions, which correspond to non-physical solutions. This support is now deprecated for the benefit of speed (about 2 to 6 times as fast now). 
- main method of calculating isoredshifts is now done by sampling the entire accretion disk space and making a contour plot of the resulting points. Algortihmically calculating the isoredshifts is still possible, but needs revision. When the isoradial corresponding to some (b, α, z) coordinate has a sharp intersection with the isoredshift (z), it needs a lot of angular precision to properly intersect and thus calculate the solution (b, α, z). I should implement something that can adaptive ly check if this intersection is indeed sharp and put angular precision where it belongs. For now, it just does not intersect and does not find solutions for all locations along some isoredshift line. 
- added gif of rotating isoredshift values for varying inclination. 
# TODO

1. Flux
  - Calculate isofluxlines in some efficiënt manner (can now be reconstructed from sampled points, but it would be neat to sample points based on isofluxlines). Perhaps calulating some points and reconstructing the lines?

2. Redshift
- revise algortihmically calculating is redshift lines 

3. General
- revise code structure. Is an isoredshift class necessary? 
- implement data classes
- implement black hole accretion disk size property (for easier plotting of ghost images) 
- add isoredshift ghost image plotting 
- add video of rotating black hole
