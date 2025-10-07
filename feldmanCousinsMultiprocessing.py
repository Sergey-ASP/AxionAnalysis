from multiprocessing import Process
import EPUtils
import gdas
from datetime import datetime, timedelta
import numpy as np
import numpy.ma as ma
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random
import copy
import pickle
import time
from scipy.constants import *

# define constants

STATION_STANDARD_DEVIATIONS = {
    'berkeley02': 2.0,
    'berkeley01': 4.0,
    'daejeon01': 0.2,
    'hayward01': 0.1,
    'krakow01': 0.8,
    'lewisburg01': 0.1,
    'mainz01': 0.1,
    'losangeles01': 1.5,
    'moxa01': 1.0,
    'oberlin01': 0.2
}

STATION_SENSITIVE_AXES = { # in form: longitude, long direction?, lattitude, lat direction?, azimuth, altitude
    'beijing01': (116.1868,'E',40.2457,'N',251,0),
    'berkeley02': (122.2570,'W',37.8723,'N',0,90),
    'canberra01': (149.1185,'E',35.2745,'S',0,90),
    'daejeon01': (127.3987,'E',36.3909,'N',0,90),
    'hayward01': (122.0539,'W',37.6564,'N',0,90),
    'krakow01': (19.9048,'E',50.0289,'N',0,90),
    'lewisburg01': (76.8825,'W',40.9557,'N',0,90),
    'losangeles01': (118.4407,'W',34.0705,'N',270,0),
    'mainz01': (8.2354,'E',49.9915,'N',0,-90),
    'moxa01': (11.6147,'E',50.6450,'N',270,0),
    'oberlin01': (82.2204,'W', 41.2950,'N',276,0),
    'belgrade01': (20.3928,'E',44.8546,'N',300,0),
    'test01': (0,'E',0,'N',0,90),
    'test02': (45,'E',30,'N',0,90),
    'test03': (120,'W',45,'N',0,90),
    'test04': (83,'E',20,'N',0,90),
    'test05': (113,'W',45,'S',0,90),
    'test06': (20,'E',17,'S',0,90),
    'test07': (0,'E',90,'N',0,90) # aligned to +z axis
}

# STATION_LIST =['test01','test02','test03','test04','test05','test06'] #'test','test','test','test','test','test','test','test']
# STATION_LIST = ['krakow01','hayward01','lewisburg01', 'mainz01', 'moxa01','daejeon01']
STATION_LIST = ['lewisburg01', 'losangeles01', 'moxa01', 'oberlin01', 'mainz01','hayward01']
BANDWIDTH_LIMIT = 100 #Hz
FREQUENCY_SAMPLING_RATE = 512 #Hz
NUMBER_COINCIDENCE = 4
# EXCESS_POWER_THRESHOLD = 

filepath = "/mnt/d/GNOMEDrive/gnome/serverdata"
start_date = "2021-08-28-00-01-00"
window_length = 2048 # seconds
# window_length = 16384 # seconds
# window_length = 32768 # seconds
end_date = EPUtils.get_end_time(start_date, window_length)
min_time_seg_length = 1 # seconds
cartesian_axes, cartesian_coords = EPUtils.coord_transform(STATION_SENSITIVE_AXES)

## END OF USER DEFINED CONSTS ##

# paralellizable FC data method
def fc_task(signal_list):
    pass
    



if __name__ == '__main__':
    ## LOAD FILES ##
    xtemp = random.uniform(-1, 1)
    ytemp = random.uniform(-1, 1)
    ztemp = random.uniform(-1, 1) 
    norm_factor = np.sqrt(xtemp**2 + ytemp**2 + ztemp**2)
    signal_vector_norm = [xtemp/norm_factor, ytemp/norm_factor, ztemp/norm_factor]   
    i_angle_offset = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
    # just load data, no injection
    sta_times,data_list, sanity_list, station_arr, starts, ends, STATION_OBJECT_LIST = EPUtils.load_data(start_date, end_date, STATION_LIST, STATION_STANDARD_DEVIATIONS, FREQUENCY_SAMPLING_RATE,
                                                                                                    filepath=filepath,
                                                                                                    shift_time=None,
                                                                                                    burst_ampl=3e14,
                                                                                                    burst_freq=17,
                                                                                                    burst_dur=256,
                                                                                                    burst_start=500,
                                                                                                    station_axes=None, # specify cartesian_axes if injecting, None else
                                                                                                    station_positions=cartesian_coords,
                                                                                                    signal_vec=signal_vector_norm,
                                                                                                    velocity=3e5,
                                                                                                    impact=0.5,
                                                                                                    i_angle=i_angle_offset,
                                                                                                    radius=6e7,
                                                                                                    verbose=True
                                                                                                    )


