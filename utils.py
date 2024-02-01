#where I'll store all the relavent fucntions for analyzing photometric and RV data
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
# import transitleastsquares as tls
import os
import sys
from uncertainties import ufloat #used for error propergation. It uses linear error propagation theory
from uncertainties import umath #documentation: https://pythonhosted.org/uncertainties/_downloads/uncertaintiesPythonPackage.pdf

#SysVels = [-2.6323, -2.6323] 
def ReWriteRV_to_Juliet(File, Trg, SysVels, out_base_name='_RV_sysCorr.dat', units_km=True, Convert2JD=2400000): # to rewrite the RV data to be in proper Juliet formate
    #File = full path to original RV data; Trg = string id of target (i.e. W6), out_base_name = the base string info for the file that will be used for all targets
    #SysVels = list of the systemic velocities calculated for each intersturment, assuming in the same order as insturment name order 
    X = np.loadtxt(File,'str') #load as string, because the last line is the insturment names
    RV, rv_err, insturments = np.array(X[:,1], dtype=np.float), np.array(X[:,2], dtype=np.float), X[:,3] 
    JD = np.array(X[:,0],dtype=np.float)+Convert2JD
    Inst, Data_groups = [],{} #to keep track of the JD, RV, and err for each insturment. Inst to keep track of the unique insturment names
    I_cnt = 0
    for I in insturments:
        if I not in Inst:
            Inst.append(I)
            Data_groups[I] = {}
            Data_groups[I]['JD'], Data_groups[I]['RV'], Data_groups[I]['rv_err'] = [JD[I_cnt]], [RV[I_cnt]], [rv_err[I_cnt]] #content is list of JDdates, RVvalues, and RV err 
        else:
            Data_groups[I]['JD'].append(JD[I_cnt]), Data_groups[I]['RV'].append(RV[I_cnt]), Data_groups[I]['rv_err'].append(rv_err[I_cnt])
        I_cnt +=1
    Path = ''
    paths = File.split('/')
    for p in range(len(paths)-1):#-1 because last element is the file name
        Path += paths[p]+'/'
    Newf = open(Path+Trg+out_base_name, 'w')
    UNITS = 1
    if units_km: #*1e3 to convert from km to meters
        UNITS = 1e3
    for i in range(len(Inst)):
        i_name = Inst[i]
        for dp in range(len(Data_groups[i_name]['JD'])):
            Newf.write(str(Data_groups[i_name]['JD'][dp])+' '+str((Data_groups[i_name]['RV'][dp]-SysVels[i])*UNITS)+' '+str(Data_groups[i_name]['rv_err'][dp]*UNITS)+' '+i_name+'\n')
    Newf.close()
    print ("Saved '"+Path+Trg+out_base_name+"'") 
    return None

def SaveTESSdat(target, SavePath=None, UseObs=None, print_options=False, JDconversion=2457000): # To quickly download TESS data and save it as .npy file for Hydra
    # to search when a target was observed by TESS https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py 
    #UseObs = list of the observations that you'd want to use to make the TESS data
    #SavePath = path and name to save file in
    print ('\033[1m '+target+' \033[0m')
    if print_options:
        print (lk.search_lightcurve(target, mission='TESS'))
    time, flux, err = np.array([]), np.array([]), np.array([])
    if UseObs:
        for uo in UseObs:
            LC_i = get_tess_lc(target, PLOT=False, obs=uo)
        time, flux, err = np.concatenate([time,LC_i.time.value+JDconversion]), np.concatenate([flux,LC_i.flux.value]), np.concatenate([err,LC_i.flux_err.value])
        np.save(SavePath,np.array([time, flux, err]))
        print ("Saved target's TESS observations ", UseObs, "as '"+SavePath+".npy'" )
    return None

def find_nearest(array, value, IDX=True): # function from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if IDX:
        return idx
    else:
        return array[idx]

def quantile(x, q, weights=None): #Compute sample quantiles with support for weighted samples. --- ripped from coner.py
    """
    Parameters
    ----------
    x : array_like[nsamples,] --- The samples.
    q : array_like[nquantiles,] --- The list of quantiles to compute. These should all be in the range ``[0, 1]``.
    weights : Optional[array_like[nsamples,]] ---  An optional weight corresponding to each sample. 
    NOTE: When ``weights`` is ``None``, this method simply calls numpy's percentile function with the values of ``q`` multiplied by 100.

    Returns
    -------
    quantiles : array_like[nquantiles,] --- The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError: For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weights``.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

def inc2b(aR, inc, ecc=0, omega=90):
    ecc_factor = (1.+ecc*np.sin(omega*np.pi/180.))/(1.-ecc**2)
    inc_factor = np.cos(inc*(np.pi/180))
    return (aR*inc_inv_factor)/ecc_factor

def b2inc(aR, b, ecc=0, omega=90):
    ecc_factor = (1.+ecc*np.sin(omega*np.pi/180.))/(1.-ecc**2)
    inc_inv_factor = (b/a)*ecc_factor
    return np.arccos(inc_inv_factor)*180./np.pi

def TransDur(P, Rs, RpRs,aRs, b=None, i=None): #calculates the transit duration, using the best fit parameters. Return value in seconds
    #expected units P[day], Rs[solar], RpRs[R_planet/R_star], aRs[semi-major/R_star] and as ufloats
    days2sec, R_sun = 86400, 6.957e8 #sec in a day, radius of sun in meters
    P_sec, Rs = P*days2sec, Rs*R_sun
    if i and not b:
        omega, ecc = 90, 0 # assuming circular orbit, and omega is in degrees
        ecc_factor = (1.0 + ecc * np.sin(omega * np.pi / 180.0)) / (1.0 - ecc ** 2) 
        inc_inv_fact = umath.cos((i*np.pi)/180) #assuming i is in degrees
        b = (inc_inv_fact/ecc_factor)*aRs
    term = umath.sqrt((Rs+(RpRs*Rs))**2 - (b*Rs)**2)/(aRs*Rs)
    return (P_sec/np.pi)*umath.asin(term) #Return value in seconds

def PhaseFold(time, Flux, FluxErr, Period, PhaseNums=1, t0=None): #phase equation from https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/folding.html
    #Note: to center phase on mid transit t0 == midtransit_time - .5Period, where midtransit_time can be ANY mid transit
    PhaseNums = int(PhaseNums) #PhaseNums needs to be an int, corresponding to the number of phases you want to fold
    if not t0:
        t0 = time[0] 
    phase = (time-t0)/(Period*PhaseNums) 
    dec = phase-np.floor(phase) #this is to properly get the decimal number of time difference over period
    phase_fold =((dec)-.5)*PhaseNums #-.5 so goes from -.5*n->0 to 0->5*n. Multiple by PhaseNums depending on how many phases I want to plot
    PhaseOrder = np.argsort(phase_fold) #need to sort for fitting purposes
    # position = np.arange(Flux.size) #to get array of indeces
    # Resort = np.argsort(position[PhaseOrder]) #To keep track of original order so I can convert back
    return phase_fold[PhaseOrder], Flux[PhaseOrder], FluxErr[PhaseOrder], PhaseOrder#, Resort #need to also keep track of the reorded time

def get_tess_lc(target, PLOT=False, obs=0):
    #obs==if target has been obseerved more than once, specify which observation you want to analize
    sr = lk.search_lightcurve(target, mission='TESS')[obs]
    lcfs = sr.download_all()
    try: #some types of data don't have PDCSAP_FLUX
        lc0 = lcfs.PDCSAP_FLUX.stitch().remove_nans()
    except: #in those cases, just load the raw data
        lc0 = lcfs
    if PLOT:
        lc0.errorbar(marker='.', alpha=0.5)
        if type(PLOT) == str:
            plt.savefig(f'{PLOT}.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'{target}_tess_lc.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    return lc0

def mask_transits(lc0, t0, p, d, PLOT=False, target='', ReturnIntransIndeces=False): #Note: t0 can be ANY mid transit time
    try: #if a lightkurve instance is passed
        time = lc0.time.value
        intransit = tls.transit_mask(time, p, d, t0)
        time, flux, flux_err = lc0.time.value[~intransit], lc0.flux.value[~intransit], lc0.flux_err.value[~intransit]
    except: #otherwise assuming a 3 element list of arrays composed of [np.array(time), np.array(flux), np.array(flux_err)]
        time = lc0[0]
        intransit = tls.transit_mask(time, p, d, t0)
        time, flux, flux_err = lc0[0][~intransit], lc0[1][~intransit], lc0[2][~intransit]
    if PLOT:
        plt.figure(figsize=(16,4))
        plt.errorbar(time, flux, yerr=flux_err, fmt='.', color='k',alpha=0.2)
        plt.ylabel("Normalized Flux")
        plt.xlabel("Time - 2457000 [BTJD days]")
        plt.savefig(f'{target}_tess_lc_transits_masked.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    if ReturnIntransIndeces:
        return time, flux, flux_err, intransit
    return time, flux, flux_err

def Readcsvdat(file, deliminator = ',', rm_bad_frames=True): #to read csv data. particularly advantageous when data is list of string and float elements
    #rm_bad_frames=to remove bad frame, which are frames with the mag, mag_err, flux, and flux_err being 99.9
    txt = open(file, 'r')
    Titles = txt.readline() #to read first line of file, which contains the string titles of data
    titles = Titles.split(deliminator)
    Data = {}
    Len, Cnt = len(titles)-1, 0
    for tit in titles:
        if Cnt == Len: #to get rid of the '\n' at the end of the last element
            Data[tit[:-1]] = []
        else:
            Data[tit] = []
        Cnt +=1
    Keys = (list(Data)) #cause py3 is stupid can't do Data.keys() to get a list anymore
    cnt = 0
    Arrys = [True]*len(Keys) #initially assuming all data can be converted to arry 
    for line in txt: 
        if cnt == 0: #first line is just names of files
            pass
        else:
            numb = line.split(deliminator)
            if rm_bad_frames:
                if numb[-2] == numb[-3] == numb[-4] == numb[-5] == '99.990':
                    continue
            for i in range(len(Keys)):
                if '\n' in numb[i]: #at the last element of line, which has a new line break
                    numb[i] = numb[i][:-1] #Want to exclude those txts
                try: # if can be converted to a float do it
                    val = float(numb[i])
                except: #else, let it be a striing
                    val = numb[i]
                    Arrys[i] = False
                Data[Keys[i]].append(val)
        cnt +=1
    for a in range(len(Arrys)): # to convert the list of floats to an np.arry
        if Arrys[a]:
            Data[Keys[a]] = np.array(Data[Keys[a]])
    return Data

def unique(list1): # function to get unique values from list  
    # intilize a null list 
    unique_list = [] 
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

def PullAppropDat(Data, Key='mag', TypeKey=None, Type=None, JDreduce=2450000, UTKeepDate=False): #To pull the data need. In the appropriate from i.e. ignore bad observations and data using a specific filter
    #to get only data of specific type. i.e. only V band observations
    #UTKeepDate, if you want to save the UTdate for future use
    if TypeKey != None and Type != None:
        Range = MatchDatType(Data, TypeKey, Type) 
        data = Data[Key][Range]
    else:
        data = Data[Key]
    
    #To include errors
    if Key == 'flux(mJy)':
        ErrKey = 'flux_err'
    else:
        ErrKey = 'mag_err'
        
    #The bad observations are given values of 99.99 for some reason. To get rid of those    
    Del = np.where(data == 99.99) 
    data = np.delete(data, Del)
    if UTKeepDate:
        UT = np.delete(np.array(Data['UT Date'])[Range], Del)
    time_clean = np.delete(Data['HJD'][Range], Del)
    time = time_clean-JDreduce
    err = np.delete(Data[ErrKey][Range], Del)
    if UTKeepDate:
        return time, data, err, UT
    return time, data, err

def MatchDatType(Data, Key, Type): #To get the data that fits a specified type given
    indices = [i for i, x in enumerate(Data[Key]) if x == Type]
    return indices

def BinDat(time, LC, LC_err, Bin=100):
    """Method to bin the data, using the weighted mean of a given databin size 'Bin' (i.e. Bin=how many datapoints in a bin, bin size).
    The error is estimated using the technique discussed here https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy"""
    Remainder = len(LC)%Bin
    LenSplit = int(np.trunc(len(LC)/Bin))
    if Remainder > 0: #To collect remainders for the last bin
        Main, mainErr, mainT= LC[:len(LC)-Remainder], LC_err[:len(LC_err)-Remainder], time[:len(time)-Remainder]
        Remain, RemainErr, remainT= LC[len(LC)-Remainder:], LC_err[len(LC_err)-Remainder:], time[len(time)-Remainder:]
        size = LenSplit+1
    else:
        Main, mainErr, mainT = LC, LC_err, time
        size = LenSplit
    BinLC, BinLCerr, BinT = np.zeros(size), np.zeros(size), np.zeros(size)
    Splits, splitErr, splitT = np.split(Main, LenSplit), np.split(mainErr, LenSplit), np.split(mainT, LenSplit)
    if Remainder: 
        Splits.append(Remain), splitErr.append(RemainErr), splitT.append(remainT)
    for S in range(len(Splits)):
        Avg, Err = weighted_avg_and_std(Splits[S], 1/(splitErr[S]**2)) #weight = 1/variance = 1/sig^2 = 1/err_bar^2
        BinLC[S], BinLCerr[S], BinT[S] = Avg, Err, np.mean(splitT[S])
    return BinT, BinLC, BinLCerr

def SigClip(Data, Key='mag', TypeKey = None, Type = None, thresh = 3, binSize = None, ReturnMod = False, JDreduce=2450000, PLOT=False, keepUT=False): 
    """To sig clip data model produced from interpolating datapoints from binned model. thresh = number of std that will be removed"""
    #to get only data of specific type. i.e. only V band observations. remove bad files. and subtract 2.45e6 from JD
    #Key = 'mag' or 'flux(mJy)'
    if type(Data) == str: #then data was given as file name that must be read in
        pulledData = True #to keep track if pulled the data from the filename
        Data = Readcsvdat(Data)
        if keepUT:
            time,data,err,UT = PullAppropDat(Data, Key=Key, TypeKey=TypeKey, Type=Type, JDreduce=JDreduce, UTKeepDate=keepUT)
        else:
            time,data,err = PullAppropDat(Data, Key=Key, TypeKey=TypeKey, Type=Type, JDreduce=JDreduce)
    else: #otherwise expecting an array of arrays where the 1st, 2nd, and 3rd elements are the flux/mag, time, and error respectively
        pulledData = False
        time,data,err = Data[0], Data[1], Data[2]
    #To use the binned values to find the 'model.' For ASAS-SN data likely won't need to do this
    if binSize:
        Err = np.ones(len(data)) #errors don't matter here
        Time, Model, Unimportant  = BinDat(time, data, Err, Bin=binSize)
        FullMod = np.interp(time, Time, Model)
    else: #if there is no obvious features in photometry, just use a flat line of mean as model. This is what we expect for an inactive star
        FullMod = np.ones(len(time))*np.mean(data)
    
    #Mark values that higher than the mean by the given threshold
    std = np.std(data)
    resid = np.abs(data-FullMod)
    Outliers = np.where(resid > std*thresh)[0]
    if PLOT:
        plt.figure(figsize = (18,7))
        if Type:
            TiTlE = 'Sigma Clipped photomitric light curve of '+Type+'_'+Key
        else:
            TiTlE = 'Sigma Clipped photomitric light curve of '+Key
        plt.title(TiTlE)
        plt.plot(time, data, 'b.', label = 'Data')
        plt.plot(time[Outliers], data[Outliers], 'r.', label = 'outliers')
        plt.plot(time, FullMod, 'k--', label = 'Model/mean')
        plt.ylabel(Key)
        plt.xlabel('JD-'+str(JDreduce))
        # plt.legend()
        # plt.show()
        # plt.close() 
    print ("sigma clipped", str(len(Outliers))+"/"+str(len(time)), "datapoints")
    time, data, err =  np.delete(time, Outliers), np.delete(data, Outliers), np.delete(err, Outliers)      
    if pulledData:
        UT = np.delete(UT, Outliers)
    if ReturnMod:
        if keepUT and pulledData:
            return time, data, err, UT, FullMod
        return time, data, err, FullMod
    else:
        if keepUT and pulledData:
            return time, data, err, UT
        return time, data, err
    
triangular_number = lambda n: int(n)*(int(n)+1)//2 #this is addition factorial. i.e X+(X-1)+(X-2)+(X-3)...

# to make a code that averages the datapoints that are close in time
def BinnedLC1(time, flux, flx_err, diff_lim=.25): # diff_lim = max difference in time [days]. 0.25 = 6 hrs (gaps longer than that are probably on different nights)
    
    #To block togther all datapoints that have similar times stamps
    DiffGroupTotal = [] #This will be a list of list, where each list holds the indeces that should be combined 
    DiffGroup_i = [] #To keep track of elements that should be combined
    Start_i, end_i = 0, 1 #The 1st index of DiffGroup_i. end_i is initalized to Start_i+1
    for d in range(0,len(time)-1):
        if time[d+1]-time[Start_i]<diff_lim: #If the 1st element (Start_i) and the last element searched (end_i) have a time difference less than 'diff_lim,' add those indeces
            if len(DiffGroup_i) == 0: #Starting new group cointaining small time differences amongst them
                DiffGroup_i.append(Start_i) #add start to the list of close times
                DiffGroup_i.append(d+1)
                Start_i = d #To keep track of where DiffGroup_i started
            else: #if DiffGroup_i already initialized, add next element to our list
                DiffGroup_i.append(d+1)
        else:
            if len(DiffGroup_i) > 1: # if the previous step was the last step in the grouped times, append the DiffGroup_i to DiffGroupTotal
                DiffGroupTotal.append(DiffGroup_i)
            if time[d+1]-time[d]> diff_lim and d not in DiffGroup_i: #only add individual element if not going to go
                DiffGroupTotal.append([d]) #in the next list
            Start_i= d+1 #reset Start_i
            DiffGroup_i = [] #rest the group list
    if len(DiffGroup_i) > 1: # For the last element
        DiffGroupTotal.append(DiffGroup_i)
    else:
        DiffGroupTotal.append([d+1])
     
    #Extra checks to make sure all the indecies are accounted for in the right sub-lists:
    TotalSum = 0 #to check if adds
    for i in range(len(DiffGroupTotal)): #To scan each sublist
        sublist = DiffGroupTotal[i]
        FirstTime, LastTime = time[sublist[0]], time[sublist[-1]]
        if LastTime-FirstTime > diff_lim: #To make sure the differences of the indeces I marked in each subdirectory are indeed within the approporate limit
            sys.exit("In sublist "+str(i)+", the 1st and last times are "+str(FirstTime)+","+str(LastTime)+". Their difference is greater than diff_lim!!!!")
        TotalSum += np.sum(sublist)
    Traiangle = triangular_number(len(time)-1) #triangle number of len(time)-1, to makes sure every index is accounted for
    if TotalSum != Traiangle:
        print ("Sum of all indeces is "+str(TotalSum)+". However, the triangle number of the inputed time array length is "+str(Traiangle)+".") 
        print ("DiffGroupTotal", DiffGroupTotal)
        sys.exit()
        
    #To weight average all blocked data
    new_time, new_flux, new_FlxErr = [], [], []
    for g in DiffGroupTotal:
        if len(g) == 1:
            new_time.append(time[g][0]),  new_flux.append(flux[g][0]), new_FlxErr.append(flx_err[g][0])
        else:
            Flux, STD = weighted_avg_and_std(flux[g], 1./(flx_err[g]**2))
            new_time.append(np.mean(time[g])),  new_flux.append(Flux), new_FlxErr.append(STD)
    return np.array(new_time), np.array(new_flux), np.array(new_FlxErr)
    # return np.array(new_time, dtype="object"), np.array(new_flux, dtype="object"), np.array(new_FlxErr, dtype="object") #need the "object" because each array is of different size

# to make a code that averages the datapoints that are close in time
def BinnedLC2(time, flux, flx_err, UTdate, diff_lim=.15): # diff_lim = max difference in time [days]. 0.25 = 6 hrs (gaps longer than that are probably on different nights)
    #To block togther all datapoints that have the same dates
    sortedIdx = np.argsort(time) #make sure everything is in chronological order
    time, flux, flx_err, UTdate = time[sortedIdx], flux[sortedIdx], flx_err[sortedIdx], UTdate[sortedIdx]
    AvgTim, AvgFlx, AvgErr = [], [], []
    cnt, TotalDP = 0, 0
    while cnt < len(UTdate): #can only run code for however many datapoints there are
        groupT, groupF, groupE, groupD = [], [], [], []
        inGroup = True
        while inGroup:
            if cnt >= len(UTdate):
                inGroup = False
                break
            yr,mth,dy = UTdate[cnt].split('-')
            dy = float(dy)
            if len(groupD) == 0: #if starting with an empty group, then the first datapoint is part of this new group
                groupD.append(UTdate[cnt]), groupT.append(time[cnt]), groupF.append(flux[cnt]), groupE.append(flx_err[cnt])
                group_yr, group_mt, group_dy = UTdate[cnt].split('-')
                group_dy = float(group_dy)
                cnt +=1
            #if on the same yr, month, day or less than 3.6 hrs apart, then part of the same observing night
            elif (yr == group_yr and mth == group_mt and np.floor(dy) == np.floor(group_dy)) or  (abs(time[cnt-1]-time[cnt])<diff_lim): 
                groupD.append(UTdate[cnt]), groupT.append(time[cnt]), groupF.append(flux[cnt]), groupE.append(flx_err[cnt])
                cnt+=1
            else:
                inGroup = False #don't continue counting, but exit while loop and start over a new loop (new group), with the same index
        Flux, STD = weighted_avg_and_std(np.array(groupF), 1./(np.array(groupE)**2))
        JDTime = np.mean(groupT)
        TotalDP += len(groupT)
        AvgTim.append(JDTime), AvgFlx.append(Flux), AvgErr.append(STD)
    if TotalDP != len(time):
        sys.exit("ERROR!!! lengths ("+str(TotalDP)+" vs. "+str(len(time))+") don't match!!!")
    return np.array(AvgTim), np.array(AvgFlx), np.array(AvgErr)

def weighted_avg_and_std(values, weights): #from: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """ Return the weighted average and standard deviation. values, weights -- Numpy ndarrays with the same shape."""
    average = np.average(values, weights=weights) #weights = 1./err_bar^2. Where err_bar=std & err_bar^2 = variance
    # variance = np.average((values-average)**2, weights=weights) # Fast and numerically precise
    variance = 1.0/np.sum(weights)
    return average, np.sqrt(variance)

# def weighted_avg_and_std(values, errs, method='1'): #from: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
#     """ Return the weighted average and standard deviation. values, weights -- Numpy ndarrays with the same shape."""
#     weights = 1./(np.array(err))**2 #weights = 1./err_bar^2. Where err_bar=std & err_bar^2 = variance
#     average = np.average(values, weights=weights) 
#     if method == '1': #this gives the variance of the obtained weighted mean
#         variance = 1.0/np.sum(weights)
#     if method == '2': #this gives the population variance
#         variance = np.average((values-average)**2, weights=weights) # Fast and numerically precise
#     return average, np.sqrt(variance)

def NormFlux(flux, time, err, method='Huitson'): 
    #to compute normalized flux. Could do either the "Huitson" method, which calculates the 'non-spotted flux'
    #or the "simple" method, which is just dividing the data by the mean
    if method.lower() == 'huitson':
        k = 1.
        sigma = np.std(flux)
        mean_flux = np.max(flux) + k*sigma
        y = flux/mean_flux #normalize flux
        yerr = err/mean_flux
    elif method.lower() == 'simple':
        mean_flux = np.mean(flux)
        y = flux/mean_flux #normalize flux
        yerr = err/mean_flux
    else:
        sys.exit("method name of '"+method+"' is not a known normalization method")
    return y, time, yerr, mean_flux #return the mean flux, so can use it for plotting purposes

def FindSigFigs(nominal, sigma, sifdig=2): # to find the significant digits in a number series. Base the significant digits on the uncertainty. The 1st 'sifdig' non-0s are the significant digits
    asymmetric = False
    if type(sigma) == list or type(sigma) == type(np.array([])): #then assumming given an upper (1st index) and lower bound (2nd index). 
        asymmetric, Sig_up, Sig_low = True, sigma[0], sigma[1]
        sigma = np.min(sigma) #then calculate the significant digits based on the smallest uncertainty
    str_cnt, sig_cnt, str_sigm = -1, 0, str(np.format_float_positional(sigma, trim='-'))
    try: #if can't find the decimal place
        whrDec = str_sigm.index('.')
    except: #then assuming it's a whole number and manually add the decimal place
        str_sigm += '.0'
        whrDec = str_sigm.index('.')
    whlNums = whrDec #this is also the length of whole numbers
    if whlNums == 1 and str_sigm[0] == '0': #except when there is a there are no whole numbers
        whlNums = whrDec-1 #then will be one off
    if whrDec > sifdig: #allow one more significant figure if the uncertainties is deep in the whole numbers
        sifdig +=1
    start_sig = False #to keep track of when to start counting significant figures, i.e. after the 1st non-0
    while sig_cnt < sifdig:
        str_cnt+=1 #wanna add the cnt 1st, so can get the proper index of that number when out of string
        if str_cnt >= len(str_sigm): #if surpassed the size of the string, before finding all the significant figures
            sig_cnt += 1 #break out, but add the extra last number for the sig figs count
            break 
        s = str_sigm[str_cnt]
        if s == '.':
            pass
        elif s == '0' and not start_sig:
            pass
        else:
            start_sig = True #as soon as you see 1 non-0 start the sig count
            sig_cnt += 1
    if sig_cnt <= whlNums: #if the significant figures are all within the whole numbers
        decimal_rnd = str_cnt+1-whrDec #then count backwards to round the proper number of significant digits (i.e. minus decimals)
        if asymmetric:
            return int(round(nominal, decimal_rnd)), int(round(Sig_up, decimal_rnd)), int(round(Sig_low, decimal_rnd))
        else:
            return int(round(nominal, decimal_rnd)), int(round(sigma, decimal_rnd))#, decimal_rnd
    else: #otherwise the significant digits are in the decimals,
        decimal_rnd = str_cnt-whrDec #and count how many digits needed after the decimal place
        if asymmetric:
            return round(nominal, decimal_rnd), round(Sig_up, decimal_rnd), round(Sig_low, decimal_rnd)
        else:
            return round(nominal, decimal_rnd), round(sigma, decimal_rnd)#, decimal_rnd

def SimgaClipping(LC, Nig=3.01,  Nneighbors=10, ExtraPars=None, update=True): 
    """sigma clipping, to get rid of the outliers. If 'Nsig'-simga further than the surrounds 'Nneighbors' points ('Nneighbors'/2 bck, 'Nneighbors'/2 front)"""
    #LC = 3 element list of light-curve info (i.e. [time, flux, flx_err]) #Nsig = how many standard deviations a given element can vary from another
    #Nneighbors = How many surrounding datapoints that will be used to calculate the standard deviation from. 
    #            Want this to be pretty low, so doesn't get confused by astrophysical features (i.e. the transit)
    #ExtraPars = dictonary of external parameters (epars as arrays). Because clipping transit data, need to also clip the external parameters so they are consistent with one another
    if Nneighbors%2: 
        Nneighbors+=1 #want Neighbors to be even, so can have equal number of datapoints infront asn behind the datapoint of interest
    if ExtraPars is not None:
        keys = list(ExtraPars.keys())
    i, TrueIdx, removed_ideces = 0, 0, []
    time, flux, err = LC[0].copy(), LC[1].copy(), LC[2].copy()
    while i < len(time):
        if i < Nneighbors/2.0: #Then at the front edge of the data and there will be more ahead points than behind points
            ahead = Nneighbors-i
            behind = Nneighbors-ahead
        elif len(time)-1 < i+Nneighbors/2.0: #Then at the back edge of the data and there will be more behind points than ahead points
            ahead = len(time)-1-i #-1 because python starts at 0
            behind = Nneighbors-ahead
        else:
            ahead, behind = Nneighbors/2.0, Nneighbors/2.0
        bck_idx, frnt_idx = int(i-behind), int(i+ahead)
        surround = np.append(flux[bck_idx:i],flux[i+1:frnt_idx+1])#get the values of the surrounding indeces, not including the actual index of interest because would skew std if it's actually on outlier
        sigma, mean = np.std(surround), np.mean(surround)
        if abs(flux[i]-mean) > Nig*sigma:
            if update:
                print ("removing timestamp:", time[i])
            time, flux, err = np.delete(time,i), np.delete(flux,i), np.delete(err,i)
            removed_ideces.append(TrueIdx) #to keep track of the original index of datapoints that have been clipped
            i = i - int(Nneighbors/2) #if removed an outlier, go back and check the other datapoints that had the outlier in the std calculation
            if ExtraPars is not None:
                for k in keys:
                    ExtraPars[k] = np.delete(ExtraPars[k],i)
        else:
            i+=1
        extras = len(np.where(np.array(removed_ideces) < TrueIdx)[0]) #indecies that have already been removed, but need to be counted for the TrueIdx counts
        TrueIdx = i+extras
        # print("\n")
    per_removed = (len(removed_ideces)/len(LC[0]))*100.0
    print ("removed "+str(len(removed_ideces))+" out of "+str(len(LC[0]))+" files. "+str(np.round(per_removed, 2))+"%")
    clippedLC = [time, flux, err]
    if ExtraPars is not None:
        return clippedLC, ExtraPars, per_removed, removed_ideces
    else:
        return clippedLC, per_removed, removed_ideces

def SimgaClippingPhased(LC, Period, Nsig=3.01, N_neighbors=20, Update=False, Plot=False, T0=None): 
    """Same routine as SimgaClipping(), but phase folds the data first, then clips the phase folded data. 
    This is done because the phase folding makes it more clear outliners relative to how the signal should be.
    Then convert from the phase folded data to the original data order"""
    phase_fold, folded_lc, folded_err, Order = PhaseFold(LC[0], LC[1], LC[2], Period, PhaseNums=1, t0=T0)
    LC_clipped, per_removed, removed_ideces = SimgaClipping([phase_fold, folded_lc, folded_err], Nneighbors=N_neighbors, update=Update)
    seq_trash = np.sort(Order[removed_ideces]) #the bad indeces in the sequential order???
    if Plot:
        FontSize = 15
        plt.figure(1, figsize=(15,4))
        plt.title('Phase folded, clipped data',fontweight='bold', fontsize=FontSize)
        plt.errorbar(LC_clipped[0], LC_clipped[1], fmt='.', color='k',alpha=0.1)# yerr=LC_clipped[2]
        phas_trash = Order[removed_ideces] #the bad indeces in the phased order
        plt.plot(phase_fold[removed_ideces], LC[1][phas_trash], 'r.')
        plt.xlabel('phase')
    #     # plt.show()
    #     # plt.close()
    used_Order = np.delete(Order, removed_ideces) #to only get useful indeces
    #To sort the time in the same order that the phased data is ordered:
    phasedtime, phasedLC, phasedErr = LC[0][used_Order], LC[1][used_Order], LC[2][used_Order]
    unphased = np.argsort(used_Order) #now revert order back to sequential order, WITHOUT outlier data
    clippedLC = [phasedtime[unphased], phasedLC[unphased], phasedErr[unphased]]
    if Plot:
        plt.figure(2, figsize=(15,4))
        plt.title('LC with clipped data',fontweight='bold', fontsize=FontSize)
        plt.errorbar(clippedLC[0], clippedLC[1], fmt='.', color='k',alpha=0.1)#yerr=clippedLC[2]
        plt.plot(LC[0][seq_trash], LC[1][seq_trash], 'r.')
        plt.xlabel('time')    
    # plt.show()
    # plt.close()
    return clippedLC, seq_trash

def UncertaintyClip(data, time, err, threshold=3): # to cut all data with uncertainties higher than 'threshold' times the average. So the really uncertain data doesn't skew the fit
    Mean_err = np.mean(err)
    good_dat = np.where(err<(threshold*Mean_err))
    data_cut, time_cut, err_cut = data[good_dat], time[good_dat], err[good_dat]
    print ("cut error:", len(data)-len(data_cut), "out of datapoints", len(data))
    return data_cut, time_cut, err_cut

def Mag2Flux(Mag, magErr, refMag=None, refFlux=1): #convert AIT relative mag to relative flux
    if not refMag: #if no reference mag given, just set the mean mag as the reference mag
        refMag = np.mean(Mag)
    pwr = (refMag-Mag)/2.51
    Flux = refFlux*(10**pwr)
    #To estimate error, calculate the flux when upper limit is added to mag and when lower limit is added to mag. Take the difference of the 2
    pwr1, pwr2 = (refMag-Mag+magErr)/2.51, (refMag-Mag-magErr)/2.51
    Err1, Err2 = abs(Flux-refFlux*(10**pwr1)), abs(Flux-refFlux*(10**pwr2))
    return Flux, np.max(np.array([Err1, Err2]), axis=0)

def Flux2Mag(flux, fluxErr, refMag=None, refFlux=1): #convert relative flux to relative mag
    if not refMag: #if no reference mag given, just set the mean mag as the reference mag
        refMag = np.mean(-2.51*np.log10(flux/refFlux))
    Mag = -2.51*np.log10(flux/refFlux) - refMag
    Mag_up = -2.51*np.log10((flux+fluxErr)/refFlux) - refMag
    Mag_dn = -2.51*np.log10((flux-fluxErr)/refFlux) - refMag
    #To estimate error, calculate the mag when upper limit is added to the flux and when lower limit is subtracted to the flux. Take the difference of the 2
    Err1, Err2 = abs(Mag_up-Mag), abs(Mag-Mag_dn)
    return Mag, np.max(np.array([Err1, Err2]), axis=0)


from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation,SkyCoord

def getLightTravelTimes(mjd, ra, dec, Tel_location, bary=True): #NEED TO CHECK IS RIGHT!!! especially with the +.5 business!!
    """
    To convert times for a target from MJD to BJD and HJD. This is code written by James McCormac (ripped from James Kirk). 
    NOTE: that the observatory must be correctly defined!
    Get the light travel times to the helio- and barycentres
    
    -----Parameters-----
    mjd: (float or array) The mjd time of observation to corrected to bjd and hjd times
    ra: (str) The Right Ascension of the target in hourangle e.g. 16:00:00
    dec: (str) The Declination of the target in degrees e.g. +20:00:00
    time_to_correct: (astropy.Time object) The time of observation to correct. The astropy.Time object must have been initialised with an EarthLocation
    Tel_location: (tuple or string) if tuple then expecting a 3 element tuple of form geocentric coordinates or longitude, latitude, and height for geodetic coordinates
                                    if string, then expecting the proper name of the observatory, so can get the proper coordinates
    bary: (boolean) if True, then return the barycentric date. Otherwise return the heliocentric date.
    ----Returns---
    ltt_bary : (TimeDelta
        The light travel time to the barycentre
    ltt_helio : TimeDelta
        The light travel time to the heliocentre
    """
    if type(Tel_location) == str: #expecting this is the observatory name, 
        OBSERVATORY = EarthLocation.of_site(Tel_location) #thus convert to geocentric coordinates
        print ("OBSERVATORY:", OBSERVATORY.geodetic)
    elif type(Tel_location) == tuple: #then don't have to do anything, cause alredy given in right formate
        OBSERVATORY = Tel_location
    else:
        sys.exit("'Tel_location', not given in expected proper format. Expecting either string of observatory name or 3d tuple of geocentric coordinates (or longitude, latitude, and height)!!!")
    time_jd = Time(mjd, format='mjd',scale='utc', location=OBSERVATORY) #The astropy.Time object must have been initialised with an EarthLocation
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    print ("target:", target)
    if bary:
        ltt_bary = time_jd.light_travel_time(target)+.5 #add the .5, because mjd is centered on the middle of the day NORMALLY. 
        print ("ltt_bary:", ltt_bary)
        time_bary = time_jd.tdb + ltt_bary
        return time_bary.value
    else:
        ltt_helio = time_jd.light_travel_time(target, 'heliocentric')+.5 
        print ("ltt_helio:", ltt_helio)
        time_helio = time_jd.utc + ltt_helio
        return time_helio.value 

# if __name__ == "__main__":
#     print (getLightTravelTimes(np.array([58207.0594593,  58207.06155952, 58207.06366692, 58207.06577258]), '20:23:29.4437', '-44:03:30.5098', (-70.40498688, -24.62743941, 2669))) #58021.64545 #0.5022521798982492 0.5022724530897286
#     path2 = '/Users/chimamcgruder/Research/GitHub/twin-planets/ASAS-SN/'
#     filename = path2+'ASASSNPhotMotW6.csv'
#     JD_V, Dat_V, Err_V, UT_V = SigClip(filename,Key='flux(mJy)',TypeKey='Filter',Type='V',PLOT=True, keepUT=True) #to identify data that's more than 3 sig from mean
#     # AvgTim_V, AvgFlx_V, AvgErr_V = BinnedLC2(JD_V, Dat_V, Err_V, UT_V)
#     # plt.errorbar(AvgTim_V, AvgFlx_V, yerr=AvgErr_V, fmt='o', mec='purple', ecolor='purple', elinewidth=1, mfc='white', ms=3)
#     plt.show()
#     plt.close()

#     JD_g, Dat_g, Err_g, UT_g = SigClip(filename,Key='flux(mJy)',TypeKey='Filter',Type='g',PLOT=True, keepUT=True) #to identify data that's more than 3 sig from mean
#     # AvgTim_g, AvgFlx_g, AvgErr_g = BinnedLC2(JD_g, Dat_g, Err_g, UT_g)
#     # plt.errorbar(AvgTim_g, AvgFlx_g, yerr=AvgErr_g, fmt='o', mec='purple', ecolor='purple', elinewidth=1, mfc='white', ms=3)
#     plt.show()
#     plt.close()