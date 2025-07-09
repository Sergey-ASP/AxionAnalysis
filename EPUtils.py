import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.scale
import matplotlib.dates as mdates
import numpy as np
import numpy.ma as ma
import gdas
import time,math,os,scipy

from datetime             import datetime
from datetime             import timedelta
from scipy                import fftpack,signal,optimize,interpolate

from gwpy.timeseries import TimeSeries,TimeSeriesList
from pycbc import psd,types,filter
import scipy.integrate
import scipy.optimize
import scipy.special
from scipy.stats import chi2, chisquare,poisson,norm


import pandas as pd
import timeit
import itertools
import calendar
import random
import copy
from random import shuffle
import seaborn as sns
import matplotlib.cm as cm
from multipledispatch import dispatch


from matplotlib import rcParams, cycler

class Station(object):
    '''
    Object for storing spectrogram and time series data for a given station.  
    Additional parameters are stored for simpler method calling.
    '''
    
    def __init__(self):
        pass
    
    def __init__(self, station, sanity, data, start_date, end_date):
        '''
        initialize station object.  Called by load_data.
        '''
        # define class attributes
        self.station = station
        self.sanity = sanity
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        
        # before summing each tile is 1x1
        self.Nf = 1
        self.Nt = 1
        self.summed = False
        self.sub = False
    
    def excess_power_init(self, sampling_rate, min_time_seg_length, bandwidth_limit, make_plot=False, save_data=True):
        '''
        calculates excess power and stores parameters in Station object.
        
        Args:
            sampling_rate (int): Sampling rate of magneti field data
            min_time_seg_length (int): TODO
            bandwidth_limit (int): Upper frequency limit of spectrogram
            make_plot (bool): Display plot of spectrogram.
            save_data (bool): save parameters and spectrograms to Station object.
        '''
        # define class attributes 
        self.sampling_rate = sampling_rate
        self.min_time_seg_length = min_time_seg_length
        self.bandwidth_limit = bandwidth_limit
        spec_normal, spec, timeseg, freq = excess_power(self.data, sampling_rate, min_time_seg_length, bandwidth_limit, make_plot)
        if save_data:
            self.spec_normal = spec_normal
            self.spec = spec
            self.time_seg = timeseg
            self.freq = freq
        return spec_normal, spec
    
    def excess_power(self, make_plot=False, save_data=True):
        '''
        Calculates excess power.  excess_power_init must be called before this method is used!
        
        Args:
            make_plot (bool): Display plot of spectrogram.
            save_data (bool): save spectrograms to Station object.
        '''
        assert all(getattr(self, attrib) for attrib in ['sampling_rate', 'min_time_seg_length','bandwidth_limit']), 'one or more excess_power attributes do not exist for current station object.'
        spec_normal, spec, timeseg, freq = excess_power(self.data, self.sampling_rate, self.min_time_seg_length, self.bandwidth_limit, make_plot)
        if save_data:
            self.spec_normal = spec_normal
            self.spec = spec
            self.time_seg = timeseg
            self.freq = freq
        return spec_normal, spec
    
    def ep_mask(self, mask, save_data=True):
        '''
        Calculates excess power by masking flagged tiles out from the average psd.
        
        Args:
            mask (ndarray): Mask of data to ignore.  non-zero values correspond result in the tile being masked.
            save_data (bool): save data to Station object
        '''
        
        # if an entire frequeny band is masked, unmask it
        freq_fix_mask = (mask.all(1).__invert__()*mask.T).T
        
        if self.summed:
            freq_fix_mask = np.kron(freq_fix_mask, np.ones((self.Nf, self.Nt))).astype(int)
        
        masked_spec = ma.array(self.spec, mask=freq_fix_mask)
        
        
        
        # calulate avg PSD
        avg_psd = np.nanmean(masked_spec, axis=1)
        
        # create matrix with avg psd replicated for each time segment
        # in the edge case where an entire frequency band is flagged, set avg to arbitrary small value
        # to make sure normalization will still flag band as above the excess power cutoff
        avg_psd_matrix = np.nan_to_num(np.outer(avg_psd, np.ones(self.spec.shape[1])))
        
        # normalize spectrogram
        normalized_spectrogram = self.spec/avg_psd_matrix
        if save_data:
            self.spec_normal = normalized_spectrogram
            
        self.add_tiles()
        return normalized_spectrogram
        
    def plot_spectrogram(self, spectrogram='excess_power'):
        '''
        Plots specified spectrogram of station.  default: excess power spectrogram.  while explicit Station attributes for \'excess_power\' do not exist, this is st
        
        Args:
            spectrogram (string): station spectrogram attribute
        '''
        # spectrogram = np.repeat(np.repeat(spectrogram,self.dt,axis=0),self.df,axis=1)
        if spectrogram == 'excess_power':
            spec = getattr(self,'spec_normal')*2
        else:
            spec = getattr(self,spectrogram)
            
        mesh = plt.pcolormesh(np.arange(spec.shape[1]), np.arange(spec.shape[0]), spec, cmap='viridis',vmin=0)
        plt.colorbar(mesh)
        plt.title('Station: {}, Spectrogram: {}'.format(self.station,spectrogram))
        plt.show()
        
    def add_tiles(self, dt=None,df=None,verbose=False):
        '''
        calculates tile summing and updates normalized spectrogram.
        
        Args:
            dt (int): width of summed tiles in seconds
            df (int): height of summed tiles in Hz
            verbose (bool): verbose messaging
        '''
        self.spec: np.ndarray
        m, n = self.spec.shape
        timerez = self.min_time_seg_length
        freqrezfromnumberofcells = 1/timerez
        
        
        arr = self.spec
        
        # if both terms are none (when tiles are summed in other internal methods) set dt and df to what they were
        # last time this method was called.
        if (dt ==None) & (df ==None):
            dt = self.dt
            df = self.df
            x = self.Nt
            y = self.Nf
        else:
            if df is None:
                y=1
                df = freqrezfromnumberofcells
            else:
                y=int(df//freqrezfromnumberofcells)
                
            if dt is None:
                x=1
                dt=timerez
            else:
                x=int(dt//timerez)
            
        delt = x*timerez
        delf = y*freqrezfromnumberofcells

        # Ensure the dimensions are divisible by dy and dx
        if df % freqrezfromnumberofcells != 0 or dt % timerez != 0:
            raise ValueError("Matrix dimensions must be divisible by tile dimensions.")

        # Reshape into blocks and sum
        reshaped = self.spec.reshape(m // y, y, n // x, x)
        spec = reshaped.sum(axis=(1, 3)) 
       
        reshaped = self.spec_normal.reshape(m // y, y, n // x, x)
        spec_normal = reshaped.sum(axis=(1, 3)) 
        
        if self.sub:
            reshaped = self.subtracted.reshape(m // y, y, n // x, x)
            self.subtracted = reshaped.sum(axis=(1, 3)) 
            
            reshaped = self.del_S.reshape(m // y, y, n // x, x)
            self.del_S = reshaped.sum(axis=(1, 3)) 
            return
            
            
            
            
        
        
        self.spec_normal_summed = spec_normal
        self.spec_summed = spec
        # self.time_seg=np.arange(dt//2,int((x+0.5)*dt),dt)
        # self.freq=np.arange(df//2,int((y+0.5)*df),df)
        self.Nf=y
        self.Nt=x
        self.dt = dt
        self.df = df
        self.summed = True
        if verbose:
            print("degrees of freedom: {}".format(2*delt*delf))
    
    def event_list(self, mask):
        '''
        returns a flattened list of event tile amplitudes from the subtracted data
        
        Args:
            mask (ndarray): mask of spectrogram that masks out event tiles
        
        '''
        # if freq is entirely flagged, unflag it
        freq_fix_mask = mask
        
        # if tile summing was done, we need to now use the tile summed non-normalized spectrogram, and the corresponding mask that goes with it
        if self.summed:
        # if False:
            
            # freq_fix_mask = np.kron(freq_fix_mask, np.ones((self.Nf, self.Nt))).astype(int)
            # generates masked spectrogram
            masked_spec =ma.array(self.spec_summed, mask=freq_fix_mask)
            
            avg_psd = np.nanmean(masked_spec, axis=1)
            subtracted = self.spec_summed-np.outer(avg_psd, np.ones(self.spec_summed.shape[1]))
            # subtracted[subtracted<0]=0
            self.sub_mask = freq_fix_mask
            self.subtracted = subtracted
            ep = masked_spec*2
            stdev = np.outer(ep.std(1), np.ones(self.spec_summed.shape[1]))
            
            # calculate inside of s unc
            del_S_sqrt = 2*stdev*self.spec_summed/np.sqrt(self.Nf*self.Nt) - stdev**2
            
            # if PSD is less than the std dev, uncertainty will just be the standard devation
            self.del_S = np.where(stdev>(self.spec_summed/np.sqrt(self.Nf*self.Nt)),stdev,np.sqrt(del_S_sqrt))      
            # self.del_S = np.sqrt(f)     
            # self.del_S = np.sqrt(f)
            print('works')
        else:
            
            # generates masked spectrogram
            masked_spec =ma.array(self.spec, mask=freq_fix_mask)
            
            # creates 
            avg_psd = np.nanmean(masked_spec, axis=1)
            subtracted = self.spec-np.outer(avg_psd, np.ones(self.spec.shape[1]))
            # subtracted[subtracted<0]=0
            self.sub_mask = freq_fix_mask
            self.subtracted = subtracted
            # self.plot_spectrogram('subtracted')
            # self.plot_spectrogram('sub_mask')
            # self.events = np.where((subtracted >0) & (masked_spec.mask), np.ones(subtracted.shape),np.zeros(subtracted.shape))
            # self.plot_spectrogram('events')
            # create array of all delta Skj values for psd array
            stdev = np.outer(masked_spec.std(1), np.ones(self.spec.shape[1]))
            
            
            # calculate inside of s unc
            del_S_sqrt = 2*stdev*self.spec - stdev**2
            
            # if PSD is less than the std dev, uncertainty will just be the standard devation
            self.del_S = np.where(stdev-self.spec>0,stdev,np.sqrt(del_S_sqrt))      
            # self.del_S = np.sqrt(f)
            
            
        masked_sub = ma.array(self.subtracted, mask=masked_spec.mask.__invert__()).compressed()    
        
        return masked_sub
        # calulate avg PSD
        
    def mean_ep(self, cutoff = 3):
        '''
        calculate the average excess power for the station.  Excludes tiles with EP above 3 sigma by default
        
        Args:
            cutoff (int): cutoff threshold in number of standard deviations
        '''
        if self.summed:
            eparr = ma.array(self.spec_normal_summed*2, mask=np.zeros(self.spec_normal_summed.shape))
        else:
            eparr = ma.array(self.spec_normal*2, mask=np.zeros(self.spec_normal.shape))
            
        stdev = np.std(eparr)
        mean = np.mean(eparr)
        ma.masked_where(eparr > mean + cutoff*stdev,eparr,False)
        return np.mean(eparr)
        
        
            
        
        
        
def _generate_noise(l, a=1., freq_samp=512):  # l = total time in seconds
    '''
    generates random noise for simulated data.
        
    Args:
        l (int): Length of generated noise in seconds
        a (int): Amplitude of noise
        freq_samp (int): sample rate of generated noise
    '''
    x = np.arange(0., float(l), 1. / freq_samp)
    y = a * np.random.normal(loc=0, size=x.size)

    gen_noise = TimeSeries(y, sample_rate=freq_samp)
    sanity_noise = TimeSeries(np.ones(int(l)), sample_rate=1.)

    return gen_noise, sanity_noise

def _generate_burst(ts_data, ampl, freq, dur, start, FSamp):
    '''
    Injects a sinusoidal burst into an existing time series.   
    
    Args:
        ts_data (TimeSeries): time series in which you want to inject the signal
        ampl (float): burst ampl [pT] 
        freq (float): burst frequency [Hz]
        dur (int): burst duration [s]
        start (int): start time for the burst [s]
        FSamp (int): Sampling rate of ts_data [Hz]
    '''
    s = ts_data.size/FSamp
    x = np.arange(0.,s,1./FSamp)
    burst = ampl*np.sin(2.*np.pi*freq*x)
    data_arrA = np.copy(ts_data.value)
    assert(start<=s), "invalid start time value for injected burst (only {} seconds of data)".format(s)
    endt = start+dur
    if endt > s:
        endt = s
    data_arrA[int(np.round(start*FSamp)):int(np.round(endt*FSamp))] = data_arrA[int(np.round(start*FSamp)):int(np.round(endt*FSamp))]+burst[0:int(np.round(dur*FSamp))] 
    
    dataBurst = TimeSeries(data_arrA, sample_rate = FSamp)
    # dataBurst = TimeSeries(data_arrA+burst)
    return dataBurst

def _generate_realistic_burst(ts_data, dir, FSamp, peak_start, n, vel, radius, station_axes, station, impact=None, freq=10, rotation=True, station_position=None, station_axis=[1,0,0]):
    '''
    Injects a realistic axion star burst into an existing time series.
    The burst is based on the Linear + Exponential solution proposed in "Approximation methods in the study of boson stars", PRD 98.
    
    NOTE: this does not acount for earth's rotation yet!
    
    Args:
        ts_data (TimeSeries): time series in which you want to inject the signal
        dir (NDArray): Direction of axion star movement
        freq (float): burst frequency [Hz]
        FSamp (int): Sampling rate of ts_data [Hz]
        peak_start (int): Time at wich the distance between the earth and axion star centers is 
            equal to the impact factor
        n (float): Number of particles. Note the injected signal amplitude is proportional to the Sqrt(n).
        vel (int): velocity of axion star [m/s]
        radius (int): Axion star radius [m]
        impact (int): Impact factor [m]
        station_axes (dict): Dictionary of station positions and sensitive axes [latitude, longitude, azimuth, altitude]
    '''
    # particle number? #
    n = 10**23
    osc_freq = freq*2*np.pi
    rad_earth = 6*10**6 
    kappa = 5.4
    # k = 4.2
    radius = 100*rad_earth
    vel = 3e5
    sigD = radius/kappa
    # compton wavelength
    compton = (3e8)/osc_freq
    k = vel/(compton*3e8)
    
    
    # Starting position r0 at time t=0 
    x0 = -2* radius
    y0 = 0 * radius
    # this is impact factor
    z0 = radius/2
    r0 = np.array([x0, y0, z0])
    
    # unit k-vector, proportional to unit velocity vector 
    khat = np.array([1,0,0])
    
    # signal_amplitude = np.sqrt(n/(7*np.pi*sigD**3))
    signal_amplitude = 10**13
    
    # Position of magnetometer as a function of time
    def r_mag(t):
        return r0+vel*khat*t
    
    def norm_r_mag(t):
        original = r_mag(t)
        return np.sqrt(original[0]**2 + original[1]**2 + original[2]**2)
    
    def r_hat(t):
        return r_mag(t)/norm_r_mag(t)
    
    # calculates gradient of Linear + Exponential sln w.r.t radius
    def grad(t):
        # return -signal_amplitude/sigD*np.exp(-norm_r_mag(t)/sigD)*np.cos(k*np.dot(khat,r_mag(t)) - osc_freq*t)*r_hat(t) - signal_amplitude*np.exp(-norm_r_mag(t)/sigD)*np.sin(k*np.dot(khat,r_mag(t))-osc_freq*t)*k*khat
        return np.multiply(-norm_r_mag(t)/sigD**2 * signal_amplitude * np.exp(-norm_r_mag(t)/sigD)*np.cos(k*np.dot(khat,r_mag(t)) - osc_freq*t),r_hat(t)) - signal_amplitude*(1+norm_r_mag(t)/sigD)*np.exp(-norm_r_mag(t)/sigD)*np.sin(k*np.dot(khat,r_mag(t)) - osc_freq*t)*k*khat
    
    
    def normgrad(t):
        original = np.nan_to_num(grad(t))
        return np.sqrt(original[0]**2 + original[1]**2 + original[2]**2) 
    
    # Earth's rotational frequency and rotational axis
    earth_freq = 2*np.pi*11.6e-6
    earth_axis = [0,0,1]
    
    def mag_dir_rotating(t):
        return station_axes*np.cos(earth_freq*t) + np.cross(station_axes,earth_axis)*np.sin(earth_freq*t) + earth_axis*np.dot(station_axes,earth_axis)*(1-np.cos(earth_freq*t))
    
    
    
    
    
    if True:
        # center of peak
        passage = 2*radius/vel
        # magnetometer sensitive axis direction
        mag_dir = np.array([1,0,0])
        
        
        def sig(t):
            return np.dot(mag_dir_rotating(t),np.nan_to_num(grad(t)))
        
        tvals = np.linspace(0,2*passage,1000)
        yvals = [sig(t) for t in tvals]
        plt.plot(tvals, yvals)
        plt.show()
    
    # Parameters:  direction, velocity, size, impact parameter
    
    pass

def load_data(start_date, end_date, station_list, std_station, freq_samp, filepath='/GNOMEDrive/gnome/serverdata/',  shift_time=None, burst_ampl=None, burst_freq=None, burst_dur=None, burst_start=None,station_axes=None,signal_vec=None):
    '''
    Loads data for specified stations within time range.  This will only load data if it is sane.
    
    Args:
        start_date (string): Earliest date.  string formatted as 'yyyy-mm-dd-HH-MM-SS' (omitted values defaulted as 0)
        end_date (string): Last date.  format same as start_date
        station_list (list[str]): List of stations.  Station names must include number identifier (e.g. 01, 02).
        std_station (dict[str, float]): Standard deviation cuttoff values for each station.  If loaded data segments have 
            standard deviations lower than the amount specified by this dictionary, those segments are marked as insane.
        freq_samp (int): Frequency sample rate.
        filepath (string): Location of files.
        shift_time (int): Time each consecutive station's start date is shifted by. There is no time shift if no value is given.
    '''
    n_stations = len(station_list)
    window_length = get_window_length(start_date, end_date)

    # initialize data and sanity lists
    stat_obj_list = []
    data_list = TimeSeriesList()
    sanity_list = TimeSeriesList()

    total_time = (datetime.strptime(end_date, '%Y-%m-%d-%H-%M-%S') - datetime.strptime(start_date, '%Y-%m-%d-%H-%M-%S')).total_seconds()

    starts = [start_date]
    ends = [end_date]
    if shift_time is not None: # Time shift applied, if given.
        start_time_dt = datetime.strptime(start_date, '%Y-%m-%d-%H-%M-%S')
        end_time_dt = datetime.strptime(end_date, '%Y-%m-%d-%H-%M-%S')

        for i in range(n_stations):
            start_time_shifted = start_time_dt + timedelta(seconds=i * shift_time)
            starts.append(start_time_shifted.strftime('%Y-%m-%d-%H-%M-%S'))

            end_time_shifted = end_time_dt + timedelta(seconds=i * shift_time)
            ends.append(end_time_shifted.strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        for _ in range(n_stations):
            starts.append(start_date)
            ends.append(end_date)

    mask = np.zeros(len(station_list))
    station_arr = np.ma.masked_array(np.array(station_list), mask)
        
    sta_times = {}
    for i, station in enumerate(station_list):
        import time
        current = time.time()
       
        try:
            _data, _sanity = _generate_noise(total_time, a=1.0, freq_samp=freq_samp)

            if station[:-2] == 'test':
                data, sanity = _generate_noise(total_time, a=1.0, freq_samp=freq_samp)
                if (station_axes != None) & (float(burst_ampl) != 0.0):
                    proj_amp = burst_ampl*np.dot(station_axes[station],np.array(signal_vec))
                    
                    data = _generate_realistic_burst(data,None,None,None,None,None,None,station,None,None,10)
                    # data = _generate_burst(ts_data=data,ampl=proj_amp,freq=burst_freq, dur=burst_dur, start=burst_start,FSamp=freq_samp)
            else:
                # use get_data_in_range() to get a TimeSeriesList of all the data between start and end
                data1, sanity1, file_list = gdas.getDataInRange(station, starts[i], ends[i], path=filepath, convert=True, sortTime=False)
                # print(data1)
                # join the individual TimeSeries in the list into a single TimeSeries, representing missing data with NaN
                data = data1.join(pad=float('nan'), gap='pad')
                sanity = sanity1.join(pad=int(0), gap='pad')

                if station in std_station:
                    station_set = True
                    min_std_dev = std_station[station]
                else:
                    station_set = False
                    print("Minimum standard deviation not set for " + station + ".")
                if station_set:
                    for x in range(1, len(data) // (window_length * int(freq_samp)) + 1):
                        check_data = data[(x - 1) * int(freq_samp) * window_length:x * int(freq_samp) * window_length]
                        std_dev = np.nanstd(check_data)
                        if std_dev.max() < min_std_dev:
                            sanity[(x - 1) * window_length:x * window_length] = [0] * window_length

                # check to make sure there are enough sane data for analysis
                assert sanity.value[np.nonzero(sanity.value)].size > 0, "no sane data for station {}".format(station)

            count = 0
            passing = False
            s = 0

            min_passing_run = 60  # at least 1 minute of continuously sane data

            while (not passing) and (s < sanity.value.size):
                if sanity[s] >= 1:
                    count += 1
                    s += 1
                    if count >= min_passing_run:
                        passing = True
                else:
                    count = 0
                    s += 1

            assert passing, "not enough sane data for analysis (< {} consecutive seconds)".format(min_passing_run)

            # flag each ±1 s of time series data with the appropriate sanity value
            data_arr = np.copy(data.value)
            sanity_arr = sanity.value
            is_sane = np.ones(data_arr.shape)
            is_sane[0:int(freq_samp)] *= sanity_arr[0]
            for pt, val in enumerate(sanity_arr[1:-1]):
                if val < 0.5:
                    is_sane[(pt) * int(freq_samp): (pt + 2) * int(freq_samp)] *= int(val)
            is_sane[-int(freq_samp):] *= sanity_arr[-1]
            is_sane = (is_sane - 1) * -1

            # double-check that missing data are flagged as insane
            is_sane[np.isnan(data_arr)] = 1

            bad = np.nonzero(is_sane)[0]

            data_arr[np.nonzero(is_sane)] = np.nan
            time = data_arr.size / freq_samp
            

            if data_arr.size < total_time * freq_samp:
                print("data size mismatch for station {}, padding end".format(station))
                data_arr = np.pad(data_arr, pad_width=(0, int(total_time * freq_samp) - data_arr.size),
                                    mode='constant', constant_values=(np.nan,)) # change 1 back to np.nan
                sanity_arr = np.pad(sanity_arr, pad_width=(0, int(total_time) - sanity.size), mode='constant',
                                    constant_values=(np.nan,))
                sanity = TimeSeries(sanity_arr, sample_rate=sanity.sample_rate)

            dtimes = np.arange(0, time, 1. / freq_samp)
            stimes = np.arange(0, time)
            data_ts = TimeSeries(data_arr, times=dtimes, sample_rate=data.sample_rate)

            data_list.append(data_ts)
            sanity_list.append(sanity)

        except AssertionError as error:
            station_arr[i] = np.ma.masked
            print(error)
        except Exception as e:
            print("Error processing station {}: {}".format(station, e))
        import time
        final = time.time()
        # sta_times[station] = final - current
        # print('successfully loaded data for station {}'.format(station))
    station_arr.data
    #create station object
    i_adjust = 0
    for i in range(station_arr.size):
        if not station_arr.mask[i]:
            station_arr.data[i]
            sanity_list[i-i_adjust]
            data_list[i-i_adjust]
            stat_obj_list.append(Station(station_list[i],sanity_list[i-i_adjust],data_list[i-i_adjust], start_date, end_date))
        else:
            i_adjust +=1
    return sta_times,data_list, sanity_list, station_arr, starts, ends, stat_obj_list

def excess_power(ts_data, sampling_rate, min_time_seg_length, bandwidth_limit, make_plot=False):
    '''
    Perform excess-power analysis on magnetic field data.
    
    Args:
        ts_data (TimeSeries): Time Series from magnetic field data
        sampling_rate (int): Sampling rate of magneti field data
        min_time_seg_length (int): TODO
        bandwidth_limit (int): Upper frequency limit of spectrogram
        make_plot (bool): Display plot of spectrogram.
    '''
    ts_data_length = np.size(ts_data)/sampling_rate
    
    nperseg =  int(min_time_seg_length*sampling_rate)
    
    # calculate spectrogram
    freq, time_seg, spectrogram = signal.spectrogram(np.array(ts_data.value), fs=sampling_rate, nperseg= nperseg, noverlap=0, scaling='spectrum', mode='psd', window='hann')

    numsampl = sampling_rate*min_time_seg_length
    # properly define PSD for spec
    spectrogram = 2*spectrogram #**2/numsampl**2
    
    # reshape spectrogram to exclude frequencies above bandwidth limit
    spectrogram = spectrogram[1:bandwidth_limit*min_time_seg_length+1,:]
    freq = freq[1:bandwidth_limit*min_time_seg_length+1]
    
    # calulate avg PSD
    avg_psd = np.nanmean(spectrogram, axis=1)
    
    # create matrix with avg psd replicated for each time segment
    avg_psd_matrix = np.outer(avg_psd, np.ones(time_seg.shape))
    
    # normalize spectrogram
    normalized_spectrogram = spectrogram/avg_psd_matrix
    
    if make_plot:
        mesh = plt.pcolormesh(time_seg, freq, normalized_spectrogram, cmap='viridis')
        plt.colorbar(mesh)
        # plt.plot(freq, spectrogram)
        # plt.yscale('log')
        plt.show()
        
    
    return normalized_spectrogram, spectrogram, time_seg, freq

def get_end_time (start_date, window_length):
    '''calculates string form of offset date using a start date and time offset (seconds)'''
    start_time= datetime.strptime(start_date, '%Y-%m-%d-%H-%M-%S')
    end_time = start_time + timedelta(seconds=window_length)
    return end_time.strftime('%Y-%m-%d-%H-%M-%S')

def get_window_length (start_date, end_date):
    '''calculates window_length (in seconds) using the start and end dates'''
    start_time= datetime.strptime(start_date, '%Y-%m-%d-%H-%M-%S')
    end_time = datetime.strptime(end_date, '%Y-%m-%d-%H-%M-%S')
    return int((end_time - start_time).total_seconds())

def _betart(x,dof,beta):
    '''internal optimization function for calculating excess power threshold'''
    return scipy.integrate.quad(chi2.pdf,x,np.inf, args=dof)[0]-beta

def _betacomb(p,Nst,nCoinA,beta):
    '''internal optimization function for calculating excess power threshold'''
    q= -beta
    for i in range(nCoinA,Nst):
        q+=scipy.special.binom(Nst,i)*(p**i)*((1-p)**(Nst-i))
    return q

def calculate_cutoff(dofA, NstA, nCoinA=4, betaA=2.7e-7, sigmaA=5.):
    '''
    Calculates the excess power above which the tail has a fractional probability of betaA assuming Gaussian noise.
    The sigma is just used to estimate the starting guess for fsolve.
    '''
    beta2=scipy.optimize.fsolve(_betacomb, betaA, args=(NstA, nCoinA, betaA))
    sol = scipy.optimize.fsolve(_betart, dofA+sigmaA,*np.sqrt(2*dofA), args=(dofA,beta2))
    return sol

def mask(threshold, station_list, coincidence_number,max_loop=4, make_plot=False, verbose=False):
    '''
    Runs mask loop to maximize sensitivity.
    Args:
        station_list (list[Station]): list of station objects
        coincidence_number (int): minimum number of stations excess power must be detected in for an event to be flagged.
        max_loop (int): maximum number of times masking loop happens.
        make_plot (bool): plots masked tiles for single iteration of loop.
        verbose (bool): print total number of new flagged events for a given excess power threshold.
    '''
    
    assert len(station_list) > 1, 'At least two stations are required'
    flag_count = 0
    mask_to_return = []
    if verbose:
        print("starting masking for ep threshold of {}".format(threshold))
    for i in range(max_loop+1):
        mask_list = []
        station: Station
        for station in station_list:
            # create mask of normalized ep, 1=masked.  make mask of summed if available.
            if station.summed:
                mask_list.append(np.nan_to_num((1)*(2*station.spec_normal_summed>threshold)))
            else:
                mask_list.append(np.nan_to_num((1)*(2*station.spec_normal>threshold)))
        
        # create mask with nonzero values for every tile that was flagged
        # by at least (coincidence_number) stations
        masksum = mask_list[0]
        for msk in mask_list[1:]:
            masksum = np.add(masksum,msk)
        mask = np.floor_divide(masksum,coincidence_number)
        
        upd_flag_count = np.count_nonzero(mask)
        
        # exit loop if no new tiles are flagged
        if upd_flag_count == flag_count or i == max_loop:
            mask_to_return = mask
            break
        else:
            if verbose:
                print(upd_flag_count - flag_count)
            flag_count = upd_flag_count
        
        for station in station_list:
            station.ep_mask(mask)
            
    mask
        
    
    if make_plot:
        
        station_list[0].mask = mask
        station_list[0].plot_spectrogram('mask')
    
    # NOTE: mask is of size of summed spectrogram.
    return flag_count/station.spec_normal.size, mask_to_return

def consistency(station_list: list[Station], mask, axes_dict: dict, ep_threshold, guess = [10,0,0],verbose = False,consistency_mask=None):
    '''
    does expected event consistency check for events that have passed the coincedence check.  
    
    Args:
        station_list (list[Station]): list of station objects
        mask (ndarray):  mask of event tiles.  1 = event.
        axes_dict (dict): dictionary of station sensitive axes.
        ep_threshold (int): excess power threshold.
        guess (list):  xyz coordinate guess for m vector.
        verbose (bool): verbose messaging.
        consistency_mask (ndarray): mask of tiles to conduct consistency check on. 1 = do check.
    '''
    
    # create lists of x and y coordinates that were flagged.  note these are indices, not actual values.
    # used to keep track of what freq and time go with each flagged tile.
    freq_fix_mask = (mask.all(1).__invert__()*mask.T).T
    
    # creates masked array: when you extract mask from this object, it converts it to bools so can invert it.
    temp_array: ma.MaskedArray
    temp_array = ma.array(freq_fix_mask,mask=freq_fix_mask)
    
    # update the mask to only include tiles that we want to calc consistency for
    if consistency_mask is not None:
        temp2arr = ma.array(freq_fix_mask,mask=consistency_mask)
        temp_array.__setmask__(temp2arr.mask & temp_array.mask)
    
    
    
    if station_list[0].summed:
        xlen, ylen = station_list[0].spec_summed.shape
    else:
        xlen, ylen = station_list[0].spec.shape
    x,y=np.meshgrid(np.arange(xlen),np.arange(ylen), indexing='ij')
    xind = ma.array(x,mask=temp_array.mask.__invert__()).flatten().compressed()
    yind = ma.array(y,mask=temp_array.mask.__invert__()).flatten().compressed()

    # create list of coincidence events in subtracted avg data
    station: Station
    event_list = [station.event_list(temp_array.mask) for station in station_list]
    
    # creates list of event vectors containing exceess power for each station. each row is an event vector S.
    event_vectors = np.vstack(event_list).T
    
    # creates list of event uncertainties for all flagged tiles at each station.  each row is the uncertanty of all stations for a given event.
    event_unc = [ma.array(station.del_S,mask=temp_array.mask.__invert__()).flatten().compressed() for station in station_list]
    event_unc = np.vstack(event_unc).T
    
    # create D matrix
    d_matrix = [axes_dict.get(station.station) for station in station_list]
    long_d = np.tile(d_matrix, len(station_list))
    
    # find mean excess power values for each station.  using cutoff of 3 sigma
    avg_ep = [station.mean_ep() for station in station_list]
    
    # create list excess power vectors for each flagged event tile, if tile summed use respective spectrogram
    if station_list[0].summed:
        ep_vectors = [ma.array(station.spec_normal_summed*2,mask=temp_array.mask.__invert__()).flatten().compressed() for station in station_list]
    else:
        ep_vectors = [ma.array(station.spec_normal*2,mask=temp_array.mask.__invert__()).flatten().compressed() for station in station_list]
    ep_vectors = np.vstack(ep_vectors).T
    # ep_vectors = ep_vectors.T
    
    
    sig_amp_list = []
    sig_vec_list = []
    x2_arr=[]
    passed_events = []
    passed_event_counter = 0
    
    for i in range(len(xind)):
        
        ## CREATE BEST FIT M VECTORS ##
        
        s = event_vectors[i]
        e = ep_vectors[i]
        prod = np.prod(s)
        # if np.prod(s) == 0.0:
        #     sig_amp_list.append(0)
        #     continue
        
        ds = event_unc[i]
        
        # chi squared function to minimize
        def x2(arr):
            return np.sum(np.square(np.square(np.dot(d_matrix,np.array(arr))) - s) / np.square(ds))
        
        # plt.plot([x2([i,0,0]) for i in np.linspace(0,60,20)])
        # plt.show()
        
        # solve for best m
        try:
            result = scipy.optimize.least_squares(x2,guess)
            m = result.x
            if not result.success:
                if verbose:
                    print("scipy optimization failed.")
                continue
        except:
            if verbose:
                print('residual error')
            m = guess
        # result = scipy.optimize.minimize(x2,guess)
        sig_vec_list.append(m)
        x2_arr.append(x2(m))
        s_fit = (np.dot(d_matrix,np.array(m)))**2
        s_adj = s/s_fit
        ds_adj = ds/s_fit
        s_avg = np.sum(s/ds**2)/np.sum(1/ds**2)
        s_adj_avg = np.sum(s_adj/(ds_adj**2))/np.sum(1/(ds_adj**2))
        ds_avg = np.sqrt(1/np.sum(1/(ds**2)))
        ds_adj_avg = np.sqrt(1/np.sum(1/(ds_adj**2)))
        
        sig_amp_list.append(s_adj_avg/ds_adj_avg)
    
        ## EXPECTED EVENT FIT CONSISTENCY CHECK ##
        
        e_fit = ((e-avg_ep)/s)*s_fit+avg_ep
        
        # check what stations are not consistently passing/failing ep threshold between e and e fit
        pass_check = True
        for i, station_ep in enumerate(e):
            if (station_ep > ep_threshold) != (e_fit[i] > ep_threshold):
                pass_check = False
                break
            
        # if the given event does not pass the consistency check, check next event
        if pass_check:
            passed_events.append(s_adj_avg/ds_adj_avg)
            passed_event_counter +=1
            
        
        
    if verbose:    
        try:
            print('{}% of events passed the consistency check ({}/{}).'.format(int(passed_event_counter*100/sig_amp_list.__len__()//1),passed_event_counter,sig_amp_list.__len__()))
        except:
            print("no events above excess power threshold.")
    return sig_amp_list, xind, yind, sig_vec_list,x2_arr,passed_events
        
    
def coord_transform(station_coords: dict):  
    '''
    Transforms station coordinates from laditude/longitude form to normalized (x,y,z).  x axis is along 0E,0N.
    
    Args:
        station_coords:  dictionary of coordinates for each station.
    '''
    
    # initialize dictionary of transformed coordinates
    transformed_coords = dict()
    
    # calculate azimuth and radial angles wrt center of sphere (radians)
    for station, coords in station_coords.items():
        match coords[1]:
            case 'W':
                phi= math.radians(-coords[0])
            case 'E':
                phi = math.radians(coords[0])
            case _:
                raise Exception("Invalid longitude for station {}. need either W or E, but was given {}.".format(station,coords[1]))
                
        match coords[3]:
            case 'N':
                theta= math.radians(coords[2])
            case 'S':
                theta = math.radians(-coords[2])
            case _:
                raise Exception("Invalid laditude for station {}. need either N or S, but was given {}.".format(station,coords[3]))
        
        # get normalized coords and angles in radians
        r = math.cos(theta)
        az = math.radians(coords[4])
        alt = math.radians(coords[5])
        x = r*math.cos(phi)
        y = r*math.sin(phi)
        z = math.sin(theta)*math.copysign(1,alt)
        
        # do coordinate transformations
        norm = np.array([x,y,z])
        true_north = np.array([-x,-y,z])
        east = np.cross(true_north,norm)
        az_vec = np.dot(true_north, math.cos(az)) + np.dot(east,math.sin(az))
        alt_vec = np.dot(az_vec,math.cos(alt)) + np.dot(norm,math.sin(alt))
        transformed_coords.update({station: alt_vec})
     
    return transformed_coords
    

    
    
        
    
    