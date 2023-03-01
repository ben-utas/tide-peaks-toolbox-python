from datetime import datetime

import numpy as np
from scipy.signal import find_peaks


def peaks(
    time: datetime,
    wl_basic: list[float],
    wl_pred: list[float],
    wl: list[float]
):
    hw_index = find_peaks(wl_basic, distance=4/24, prominence=0.1)
    hw = wl_basic[hw_index]
    hw_time = time[hw_index]

    lw_index = find_peaks(-wl_basic, distance=4/24, prominence=0.1)
    lw = -wl_basic[hw_index]
    lw_time = time[hw_index]

    bhw = hw
    bhw_time = hw_time
    blw = lw
    blw_time = lw_time

    basic_hw = {'time': bhw_time, 'hw': bhw}
    basic_lw = {'time': blw_time, 'lw': blw}

    pred_hw = basic_hw
    pred_lw = basic_lw
    obs_hw = basic_hw
    obs_lw = basic_lw

    hw_index = find_peaks(wl_pred, distance=0, prominence=0)
    hw = wl_basic[hw_index]
    hw_time = time[hw_index]

    lw_index = find_peaks(-wl_pred, distance=0, prominence=0)
    lw = -wl_basic[lw_index]
    lw_time = time[lw_index]

    for i in range(len(bhw)):
        idx = np.where(hw_time > bhw_time[i]-4/24 & hw_time < bhw_time[i]+4/24)
        if idx:
            peaks = hw[idx]
            high_water = max(peaks)
            t = hw_time[idx]
            t = t[peaks == high_water]
            pred_hw['hw'][i] = high_water
            pred_hw['time'][i] = t[0]
        else:
            pred_hw['hw'][i] = np.nan
            pred_hw['time'][i] = np.nan
