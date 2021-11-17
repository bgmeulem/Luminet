# Luminet
This repo provides material for recreating the famous paper by Jean-Pierre Luminet, simulating the first image of a black hole. This is done in Python 3.8

# Latest updates:
Rewrite of entire branch. Apologies for the lack of incremental updates, it seems my upstream branch has not been set up properly. This is fixed now.
Everything has been moved to two files: black_hole_math.py (for calculating equations and variables) and a file containing the class 
BlackHole() for easy access to calculations.

Vastly improved speed and stability. Impact parameters are calculated using the midpoint method: stable and quick.
Added functionality to calculate redshifts and isoredshift lines

# TODO / Bugs
It seems the calculation of redshifts is not correct yet. It finds values that are too extreme compared to the paper. Unsure about why this is.
