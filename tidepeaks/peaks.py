from datetime import datetime

import numpy as np
from scipy.signal import find_peaks


def peaks(
    time: list[datetime],
    wl_basic: list[float],
    wl_pred: list[float],
    wl: list[float]
):
    """
    Identifies high water (HW) peaks and low water (LW) troughs in tide gauge data.

    NOTES:
    - High frequency peaks/troughs (e.g., wind waves and seiches) are excluded by the minimum separation between peaks of 4 hours (distance = 4/24).
    - The minimum height difference between consecutive peak and trough is is set to 10 cm (prominence = 0.1).
    - For further information refer to the Tide Peaks Toolbox User Manual.

    BEFORE EXECUTING THIS FUNCTION:
    1. Execute regdata function to interpolate raw data to regular intervals, remove duplicates, and fill gaps with NaNs.
    2. Execute utide functions to compute harmonic coefficients and model basic and predicted tide.

    Args:
        time (list[datetime]): Regular time list in datetime format.
        wl_basic (list[float]): Basic water levels modelled from 8 most influential constituents.
        wl_pred (list[float]): Predicted water levels modelled from all constituents.
        wl (list[float]): Observed water levels in metres.

    Returns:
        basic_hw, basic_lw, pred_hw, pred_lw, obs_hw, obs_lw (dicts): 'time' and 'hw'/'lw' keys returning arrays of basic, predicted, and observed heights, respectively.
    """
    # Locate tidal maximas in basic tide model
    hw_index = find_peaks(wl_basic, distance=4/24, prominence=0.1)
    hw = wl_basic[hw_index]
    hw_time = time[hw_index]

    # Locate tidal minimas on inverted model
    lw_index = find_peaks(-wl_basic, distance=4/24, prominence=0.1)
    lw = wl_basic[hw_index]
    lw_time = time[hw_index]

    # Copy basic peaks to new variables
    bhw = hw
    bhw_time = hw_time
    blw = lw
    blw_time = lw_time

    # Output to dicts
    basic_hw = {'time': bhw_time, 'hw': bhw}
    basic_lw = {'time': blw_time, 'lw': blw}

    # Pre-allocate tables for predictions and observations
    pred_hw = basic_hw
    pred_lw = basic_lw
    obs_hw = basic_hw
    obs_lw = basic_lw

    # Use findpeaks to locate tidal maximas in predicted tide model
    hw_index = find_peaks(wl_pred, distance=0, prominence=0)
    hw = wl_pred[hw_index]
    hw_time = time[hw_index]

    # Locate tidal minimas on inverted predicted model
    lw_index = find_peaks(-wl_pred, distance=0, prominence=0)
    lw = wl_pred[lw_index]
    lw_time = time[lw_index]

    # Match basic model peaks to predicted maxima/minima
    for i in range(len(bhw)):
        index = np.where(
            hw_time > bhw_time[i]-4/24 &
            hw_time < bhw_time[i]+4/24
        )
        if index:
            peaks = hw[index]
            high_water = max(peaks)
            t = hw_time[index]
            t = t[peaks == high_water]
            pred_hw['hw'][i] = high_water
            pred_hw['time'][i] = t[0]
        else:
            pred_hw['hw'][i] = np.nan
            pred_hw['time'][i] = np.nan
    for i in range(len(blw)):
        index = np.where(
            lw_time > blw_time[i]-4/24 &
            lw_time < blw_time[i]+4/24
        )
        if index:
            peaks = lw[index]
            low_water = min(peaks)
            t = lw_time[index]
            t = t[peaks == low_water]
            pred_lw['lw'][i] = low_water
            pred_lw['time'][i] = t[0]
        else:
            pred_lw['lw'][i] = np.nan
            pred_lw['time'][i] = np.nan

    # Locate tidal maximas in basic tide model
    hw_index = find_peaks(wl, distance=0, prominence=0)
    hw = wl[hw_index]
    hw_time = time[hw_index]

    # Locate tidal minimas on inverted model
    lw_index = find_peaks(-wl, distance=0, prominence=0)
    lw = wl[lw_index]
    lw_time = time[lw_index]

    # Match basic model peaks to observed maxima/minima
    for i in range(len(bhw)):
        index = np.where(
            hw_time > bhw_time[i]-4/24 &
            hw_time < bhw_time[i]+4/24
        )
        if index:
            peaks = hw[index]
            high_water = max(peaks)
            t = hw_time[index]
            t = t[peaks == high_water]
            obs_hw['hw'][i] = high_water
            obs_hw['time'][i] = t[0]
        else:
            obs_hw['hw'][i] = np.nan
            obs_hw['time'][i] = np.nan
    for i in range(len(blw)):
        index = np.where(
            lw_time > blw_time[i]-4/24 &
            lw_time < blw_time[i]+4/24
        )
        if index:
            peaks = lw[index]
            low_water = min(peaks)
            t = lw_time[index]
            t = t[peaks == low_water]
            obs_lw['lw'][i] = low_water
            obs_lw['time'][i] = t[0]
        else:
            obs_lw['lw'][i] = np.nan
            obs_lw['time'][i] = np.nan
    return basic_hw, basic_lw, pred_hw, pred_lw, obs_hw, obs_lw
