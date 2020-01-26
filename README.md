# Luminet
This repo provides material for recreating the famous paper by Jean-Pierre Luminet, simulating the first image of a black hole. This is done in Python 3.6

## Todo:
- Use previous solution as initial guess for next convergence? Try different solvers, search whithin a certain interval (check paper) might keep isoredshift lines from going to infinity. Plot redshift at different angles from R=0 -> ~60 R to see behaviour.
- Calculate gamma according to equation (11)
- Use this to make ghost images (or higher order images)
- Calculate redshifts according to equation (19), plot isoredshiftlines
- Calculate flux and plot isofluxlines
- How to do flux lines? Center of these loops are in general not at R=0, so looping the angle gives two solutions in general. Fix R=R_center? Coördinate transformation to complex plane?
- Monte Carlo simulate some points in (r, alpha) space. Assign P (thus b) and redshift values to each point. Plot (r, b) with redshift color overlayed, as well as flux. Alternatively, make each point white and make alpha inversely proportional to redshift and proportional to flux to more closely simulate original photograph.
