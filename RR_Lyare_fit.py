import numpy as np
import matplotlib.pyplot as plt
import corner
import os
import sys
sys.path.append('../')
import utils
from math import factorial
# import pymultinest

#INPUT
TickFont, LabFont = 15, 20
SavePath = '/Users/chimamcgruder/Research/TwinPlanets/Juliet_fitting/HATS29/'
X = np.load('H29_LC.npy') #the sigma clipped unphased data
Time, Mag, Err = X[0,:], X[1,:], X[2,:]
H29_t0, H29_p, H29_d = 2458657.83252, 4.606184, 1.2*(3.121643/24)
plt.figure(0, figsize=(15,5))
plt.plot(Time-2.4586e6, Mag, 'k.')
plt.ylabel('Relative Flux', fontweight='bold', fontsize=LabFont)
plt.xlabel('BJD - 2.4586e6 (days)', fontweight='bold', fontsize=LabFont)
plt.xticks(fontweight='bold', fontsize=TickFont)
plt.yticks(fontweight='bold', fontsize=TickFont)
plt.savefig(SavePath+'TESSdata_OG.png',bbox_inches='tight')
print ("Saved 'TESSdata_OG.png' in path: '"+SavePath+"'")
OOTtime, OOTf, OOTsigma = utils.mask_transits([Time, Mag, Err],H29_t0, H29_p, H29_d)
RR_Lyrae_Period = 0.631343024449301 #derived from template matching
f_time, f, sigma, PhseOrdr = utils.PhaseFold(OOTtime, OOTf, OOTsigma , RR_Lyrae_Period, t0=H29_t0)
f_time_binned, f_binned, sigma_binned = utils.BinDat(f_time, f, sigma, Bin=100)
ndata, n_params, n_live_points, mean_sigma_binned = len(f_time_binned), 2, 1000, np.mean(sigma_binned)
# out_file = 'SmoothingFit1/'
# if not os.path.exists(out_file[:-1]):
#     os.mkdir(out_file[:-1])

chi2 = lambda O, E: np.sum((O-E)**2/E)

def savitzky_golay(y, window_size, order, deriv=0, rate=1): #source: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    ### Parameters
    --------------
    y : array_like, shape (N,) == the values of the time history of the signal.
    window_size : int == the length of the window. Must be an odd integer number.
    order : int === the order of the polynomial used in the filtering. Must be less then `window_size` - 1.
    deriv: int == the order of the derivative to compute (default = 0 means only smoothing)
    ### Returns
    -----------
    ys : ndarray, shape (N)== the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688 """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#CAN'T DO THIS BECAUSE WILL FIT EACH WIGGLE IN THE DATA!
# def FindNearestOdd(Y):
#     nearestInt = np.round(Y,0)
#     if nearestInt % 2 != 1: #if the nearest int isn't an odd number, then find the nearest odd number
#         nearestOdds = np.array([nearestInt-1, nearestInt+1])
#         nearOdd = np.argmin(np.absolute(Y-nearestOdds))
#         return int(nearestOdds[nearOdd])
#     else: #otherwise return the int
#         return int(nearestInt)

# def quantile(x, q, weights=None): #Compute sample quantiles with support for weighted samples. --- ripped from coner.py
#     """
#     Parameters
#     ----------
#     x : array_like[nsamples,] --- The samples.
#     q : array_like[nquantiles,] --- The list of quantiles to compute. These should all be in the range ``[0, 1]``.
#     weights : Optional[array_like[nsamples,]] ---  An optional weight corresponding to each sample. 
#     NOTE: When ``weights`` is ``None``, this method simply calls numpy's percentile function with the values of ``q`` multiplied by 100.

#     Returns
#     -------
#     quantiles : array_like[nquantiles,] --- The sample quantiles computed at ``q``.

#     Raises
#     ------
#     ValueError: For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weights``.
#     """
#     x = np.atleast_1d(x)
#     q = np.atleast_1d(q)
#     if np.any(q < 0.0) or np.any(q > 1.0):
#         raise ValueError("Quantiles must be between 0 and 1")

#     if weights is None:
#         return np.percentile(x, list(100.0 * q))
#     else:
#         weights = np.atleast_1d(weights)
#         if len(x) != len(weights):
#             raise ValueError("Dimension mismatch: len(weights) != len(x)")
#         idx = np.argsort(x)
#         sw = weights[idx]
#         cdf = np.cumsum(sw)[:-1]
#         cdf /= cdf[-1]
#         cdf = np.append(0, cdf)
#         return np.interp(q, cdf, x[idx]).tolist()

# #===== TRANSFORMATION OF PRIORS:
# transform_uniform = lambda x,a,b: int(np.round(a + (b-a)*x, 0))
# transform_uniform_odds = lambda x,a,b: FindNearestOdd(a+(b-a)*x)

# def prior(cube, ndim, nparams): #assuming cube goes from 0 to 1, need to map it to the desired prior range I want
#     # Prior on the polynomial order fit from 3 to 13, but only integers:
#     cube[0] = transform_uniform(cube[0],3.,13.)

#     # Prior on window size uniform can go from to 5 to 81, but only odd integers and has to be 2 greater than poly order:
#     cube[1] = transform_uniform_odds(cube[1],40.,81)

# def loglike(cube, ndim, nparams):
#     try: #a lot of the time the window/order combination doesn't work. 
#         model = savitzky_golay(f_binned, cube[1], cube[0])
#     except: #In that case, give 0 likelyhood for those parameters
#         return -np.inf
#     # normalisation
#     norm = ndata*(-np.log(2*np.pi) - np.log(mean_sigma_binned*2)) 
#     # chi-squared
#     chisq = np.sum((f_binned-model)**2)/(mean_sigma_binned**2)
#     return 0.5*(norm - chisq)

# if not os.path.exists(out_file+'posteriors_trend_george.npy'):
#     pymultinest.run(loglike, prior, n_params, n_live_points=n_live_points, outputfiles_basename=out_file, resume=False, verbose=True)
#     # Get output:
#     output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params=n_params)
#     # Get out parameters: this matrix has (samples,n_params+1):
#     posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
#     unstack_files = np.array([posterior_samples[:,0], posterior_samples[:,1]]) #same as POST, but without labeling each variable
#     np.save(out_file+'posteriors_trend_george', unstack_files)
# else:
#     print ("Posterior samples already found. Using those...")
#     unstack_files = np.load(out_file+'posteriors_trend_george.npy')

# quntL, mean, quntH = 0.15865,0.50, 0.84135 #upper and lower bound of 1sig and mean value
# Labels = ['order', 'window'] #to keep track of the names, since POST doesn't preserve order
# Means = []
# print ("\n\n\nBest fits:")
# for l in range(len(Labels)):
#     Lower, Mean, Upper = quantile(unstack_files[l,:],[quntL, mean, quntH])
#     Upper, Lower = Upper-Mean, Mean-Lower
#     Means.append(Mean)
#     print (Labels[l]+' = '+str(Mean)+' +'+str(Upper)+'-'+str(Lower))


# #plot corner plot:
# postsamples = np.vstack((unstack_files)).T
# fig = corner.corner(postsamples, labels=Labels, quantiles = [quntL, mean, quntH])
# fig.savefig(out_file+'/Corner.png')
# plt.close()
#CAN'T DO THIS BECAUSE WILL FIT EACH WIGGLE IN THE DATA!

# ###To plot the phase folded mag data and its best fit
# Wind, Ord = FindNearestOdd(Means[1]), np.round(Means[0],0)
Wind, Ord = 51.0, 10.0 #this was emperically found to track the data the best, but also smooth it out the most
#To add the last bin to the begining of the array and the first bin to the end of the array so the smoothing works better:
front_f, front_t = f_binned[-3:], f_time_binned[-3:]-1
back_f, back_t = f_binned[:3], f_time_binned[:3]+1
extra_f_bin, extra_time =  np.append(front_f, f_binned), np.append(front_t, f_time_binned)
extra_f_bin, extra_time =  np.append(extra_f_bin, back_f), np.append(extra_time, back_t)
fit = savitzky_golay(extra_f_bin, Wind, Ord)
Chi2 = chi2(fit, extra_f_bin)
print ("model used window, order, and chi^2:", Wind, Ord, Chi2)
phTime, phMag, phErr, phOrdr = utils.PhaseFold(Time, Mag, Err, RR_Lyrae_Period, t0=H29_t0)
FIT = np.interp(phTime, extra_time, fit)
sigma_binned2, FIT2, F2 = np.append(sigma_binned, sigma_binned), np.append(FIT, FIT), np.append(f,f)#To plot 2 phases
f_time2, f_time_binned2, f_binned2 = np.append(f_time, f_time+1), np.append(f_time_binned, f_time_binned+1), np.append(f_binned,f_binned)
phTime2 = np.append(phTime, phTime+1)
plt.figure(13, figsize=(15,5))
plt.plot(f_time2, F2, 'k.', markersize=1)
plt.errorbar(f_time_binned2, f_binned2, yerr=sigma_binned2, marker='o', mfc='white', color='blue', ls='--', zorder=6, markersize=2, label='binned data')
plt.plot(phTime2, FIT2, 'r-', zorder=7, label='RR-Lyare fit') #plt.plot(f_time_binned, fit, 'r-', zorder=7)
plt.ylim([np.max(F2), np.min(F2)])
plt.xlim([-.5,1.5])
plt.ylabel('Relative Mag', fontweight='bold', fontsize=LabFont)
plt.xlabel('Phase', fontweight='bold', fontsize=LabFont)
plt.legend(prop={'size':LabFont, 'weight':'bold'})
plt.xticks(fontweight='bold', fontsize=TickFont)
plt.yticks(fontweight='bold', fontsize=TickFont)
plt.savefig(SavePath+'TESS_RRLyarePhase.png',bbox_inches='tight')
print ("Saved 'TESS_RRLyarePhase.png' in path: '"+SavePath+"'")

#To correct data for RR-Lyrae signal
Phaz, MagPhaz, ErPhaz, Ordr = utils.PhaseFold(Time, Mag, Err, RR_Lyrae_Period, t0=H29_t0)
corrFluxPhaz = (MagPhaz/FIT)  #this includes the transit data
corrFluxPhaz = corrFluxPhaz + (np.mean(FIT)-np.mean(corrFluxPhaz)) #to recenter data at right magnitude

### To unphase the data, rephase with the transit period, and do sigma clipping there
unphase = np.argsort(Ordr)
clippedLC, removed = utils.SimgaClippingPhased([Time, corrFluxPhaz[unphase], ErPhaz[unphase]], H29_p, Plot=True, T0=H29_t0+(H29_p/2))

### To back convert to magnitude
print ("np.mean(clippedLC[2])", np.mean(clippedLC[2]))
# FinalFlux, FinalErr = utils.Mag2Flux(clippedLC[1], clippedLC[2], np.mean(clippedLC[1]))
plt.figure(3, figsize=(15,5))
plt.plot(clippedLC[0], clippedLC[1], 'k.', markersize=1, alpha=.5)


### To plot the phase folded corrected data (in terms of flux)
plt.figure(4, figsize=(15,5))
P_final, t0_final = 4.6058826873, 2457031.9566611894 # best parameters obtained from final fit: 
Rs = 1.066 #'/Users/chimamcgruder/SSHFS/CM_GradWrk/Juliet_jointFits/HATS29/H29_fulljointfit_fixE_iter1/posteriors.dat'
RpRs, B, A = 0.1132296239, 0.3946695679, 11.3574759801
dur = utils.TransDur(P_final, Rs, RpRs, A, b=B)
print ("dur:", dur/(60*60), '[hr]')
InTransPhase = ((dur*1.05)/(60*60*24))/P_final #to determine what fraction of the phase is in transit (with a lil buffer)
#TESS corrected data
fFPhaz, fFlxPhaz, fFErrPhaz, fFOrdr = utils.PhaseFold(clippedLC[0], clippedLC[1], clippedLC[2], P_final, t0=t0_final+(P_final/2))
#To renormalize the data by taking the mean of the OOT data
OOT_bck, OOT_frnt = np.where(-(InTransPhase/2)>fFPhaz)[0], np.where(fFPhaz>(InTransPhase/2))[0]
OOT = np.concatenate((OOT_bck, OOT_frnt))
meanPhazFlux = np.mean(fFlxPhaz[OOT])
# IT = np.where((-(InTransPhase/2)<fFPhaz)&(fFPhaz>(InTransPhase/2)))[0]
# plt.axvspan(fFPhaz[IT][0], fFPhaz[IT][-1], color='r', alpha=.5)
fFlxPhaz, fFErrPhaz = fFlxPhaz/meanPhazFlux, fFErrPhaz/meanPhazFlux
fFPhaz_binned, fFlxPhaz_binned, fFErrPhaz_binned = utils.BinDat(fFPhaz, fFlxPhaz, fFErrPhaz, Bin=100)
plt.plot(fFPhaz, fFlxPhaz, 'c.', markersize=1, alpha=.5)
plt.errorbar(fFPhaz_binned, fFlxPhaz_binned, yerr=fFErrPhaz_binned, marker='o', mfc='white', color='blue', ls='', zorder=6, markersize=2, label='TESS')
#HATS data
Y = np.load(SavePath+'H29_HATSsouthFinal.npy')
SOUTH_Time, SOUTH_Flx, SOUTH_Err = Y[0,:], Y[1,:], Y[2,:]
SOUTH_phTime, SOUTH_phFlx, SOUTH_phErr, SOUTH_phOrdr = utils.PhaseFold(SOUTH_Time, SOUTH_Flx, SOUTH_Err, P_final, t0=t0_final+(P_final/2))
#To renormalize the data by taking the mean of the OOT data
OOT_bck, OOT_frnt = np.where(-(InTransPhase/2)>SOUTH_phTime)[0], np.where(SOUTH_phTime>(InTransPhase/2))[0]
OOT = np.concatenate((OOT_bck, OOT_frnt))
meanPhazFlux = np.mean(SOUTH_phFlx[OOT])
SOUTH_phFlx, SOUTH_phErr = SOUTH_phFlx/meanPhazFlux, SOUTH_phErr/meanPhazFlux
SOUTH_fFPhaz_binned, SOUTH_fFlxPhaz_binned, SOUTH_fFErrPhaz_binned = utils.BinDat(SOUTH_phTime, SOUTH_phFlx, SOUTH_phErr, Bin=100)
plt.plot(SOUTH_phTime, SOUTH_phFlx, 'm.', markersize=1, alpha=.5)
plt.errorbar(SOUTH_fFPhaz_binned, SOUTH_fFlxPhaz_binned, yerr=SOUTH_fFErrPhaz_binned, marker='o', mfc='white', color='red', ls='', zorder=6, markersize=2, label='HATSouth')
plt.ylim((0.978, 1.015))
plt.xlim((-.2, .2))
plt.ylabel('Relative Flux', fontweight='bold', fontsize=LabFont)
plt.xlabel('Phase', fontweight='bold', fontsize=LabFont)
plt.legend(prop={'size':LabFont, 'weight':'bold'})
plt.xticks(fontweight='bold', fontsize=TickFont)
plt.yticks(fontweight='bold', fontsize=TickFont)
# plt.plot(fFPhaz_binned, np.ones(len(fFPhaz_binned))*np.min(fFlxPhaz_binned), 'r--')
plt.savefig(SavePath+'TESS_Transit_Phase.png',bbox_inches='tight')
print ("Saved 'TESS_Transit_Phase.png' in path: '"+SavePath+"'")
plt.show()
plt.close()

#To save the TESS data after the RR Lyrae signal has been corrected for
path = "/Users/chimamcgruder/Research/TwinPlanets/Juliet_fitting/HATS29/"
np.save(path+"H29_TESS", np.array(clippedLC))
print ("Saved final LC as 'H29_TESS.npy' in '"+path+"'")
