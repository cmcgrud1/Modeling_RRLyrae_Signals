# Modeling_RRLyrae_Signals
To model contamination from a background RR Lyrae star in TESS data

Given that the TESS observations have a large field of view, a lot of stars can contaminate the data of your star of interest. In general, this is not a huge problem, given that TESS detects plants by variation of light due to a planet passing in front of its host star. Since, stars' variation is marginal compared to the signal caused by a transiting planet, the overall effect of a background star doesn't hinder in the measurements of the passing planet. However, for the case of HATS-29b, there was an RR Lyrae star in the background. The variation of the RR Lyrae star had a frequency higher than that of the planet transit and a signal multiple times greater. Thus, the features of the planet transit could not be model until the RR Lyrae background signal was modeled out. This code does just that, by first modeling the features (period, amplitude, and asymmetric structure) of the RR Lyrae signal. Then it subtracts out that signal allowing for analysis of the transit of interest.

This routine was implemented and utilized in McGruder et. al. 2022 (ADS link: [https://ui.adsabs.harvard.edu/abs/2023ApJ...944L..56M/abstract](url))
Required packages (excluding anaconda standard packages):
1) corner - [https://github.com/dfm/corner.py](url)
2) utils.py - This is an auxiliary module developed in house, because the routines in here are used by multiple scripts. This is the same utils.py file used in [https://github.com/cmcgrud1/JointParameterEstimation](url)

Input data (included):
1) "H29_HATSsouthFinal.npy" is the ground-based data of HATS-29b, from HATSsouth. Used to compare against the corrected TESS data.
2) "H29_LC.npy" is the TESS data coming from the TESS reduction pipeline
3) "H29_TESS.npy" is the final TESS corrected data outputted by this script


Output of code (included): 
1) "TESSdata_OG.png" is the TESS data that has only be edited with the TESS extraction pipeline. 
2) "TESS_RRLyarePhase.png" is the isolated, phase folded RR Lyrae signal
3) "TESS_Transit_Phase.png" is the phase folded TESS data after the RR Lyrae signal was modeled out. Additional HATSsouth ground-based data is included. This data didn't have contamination because the field of view was much smaller, allowing only the star of interest to be observed.
