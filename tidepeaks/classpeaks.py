import numpy as np
from scipy.signal import find_peaks


def classpeaks(
    basic_hw: dict,
    basic_lw: dict,
    measured_hw: dict,
    measured_lw: dict
):
    """
    Identifies high water peaks and low water troughs in tide data

    NOTES:
    - Classifies higher high water (HHW) and lower low water (LLW) from identified HW/LW
    - Values are given as time (datetime) and height (m)
    - Diurnal peaks are identified from the maximum peak and minimum trough in each sequence of two semidiurnal cycles, starting from the first crossing of Z0. Predicted HW/LW are identified from the 'Basic' tide fit using only the 8 most influential constituents (wl_basic) (this reduces misclassifications in complex tidal regimes)
    - For further information refer to the Tide Peaks Toolbox User Manual

    Author: Karen Palmer, University of Tasmania
    Author: Ben Mildren, University of Tasmania
    Created: 01/03/2023

    Args:
        basic_hw (dict): Dictionary of basic high water heights with 'time' keys storing datetime values and 'hw' keys storing floats.
        basic_lw (dict): Dictionary of basic high water heights with 'time' keys storing datetime values and 'lw' keys storing floats.
        measured_hw (dict): Dictionary of measured high water heights with 'time' keys storing datetime values and 'hw' keys storing floats.
        measured_lw (dict): Dictionary of measured low water heights with 'time' keys storing datetime values and 'lw' keys storing floats.

    Returns:
        hhw, llw, hws, lws (dicts): 'time' keys storing datetime values, and 'hhw', 'llw', 'hws', 'lws' keys respectively storing floats of higher high water, lower low water, higher semidiurnal water, and lower semidiurnal water. 
    """
    time_hhw = np.empty(len(basic_hw['time']))
    hhw = np.empty(len(basic_hw['hw']))
    time_llw = np.empty(len(basic_lw['time']))
    llw = np.empty(len(basic_lw['lw']))

    for i in range(0, len(basic_hw['hw'], 2)):
        hw = [basic_hw['hw'][i], basic_hw['hw'][i+1], basic_hw['hw'][i+2]]
        t = [basic_hw['time'][i], basic_hw['time'][i+1], basic_hw['time'][i+2]]
        if t[1] - t[0] < 16/24:
            chosen_hhw = max(hw[0:1])
            chosen_t = t[hw == chosen_hhw]
            hhw[i] = chosen_hhw
            time_hhw[i] = chosen_t[0]
        elif (
            t[1] - t[0] >= 16/24 and
            t[1] - t[0] < 32/24 and
            t[2] - t[1] >= 16/24 and
            t[2] - t[1] < 32/24
        ):
            chosen_hhw = hw[0:1]
            chosen_t = t[0:1]
            hhw[i:i+1] = chosen_hhw
            time_hhw[i:i+1] = chosen_t
        elif (
            t[1] - t[0] >= 16/24 and
            t[1] - t[0] < 32/24 and
            t[2] - t[1] < 16/24
        ):
            chosen_hhw = min(hw)
            if chosen_hhw == hw[2]:
                chosen_hhw = hw[0:1]
                chosen_t = t[0:1]
                hhw[i:i+1] = chosen_hhw
                time_hhw[i:i+1] = chosen_t
            else:
                hhw[i] = hw[0]
                time_hhw[i] = t[0]
        else:
            hhw[i] = np.nan
            time_hhw[i] = np.nan

    while hhw.count(np.nan):
        hhw.remove(np.nan)
    while time_hhw.count(np.nan):
        time_hhw.remove(np.nan)

    basic_hhw = hhw
    basic_time_hhw = time_hhw

    for i in range(len(hhw)):
        index = np.where(
            measured_hw['time'] > basic_time_hhw[i] - 8/24 &
            measured_hw['time'] > basic_time_hhw[i] + 8/24
        )
        if index:
            peak = measured_hw['hw'][index]
            chosen_hhw = max(peak)
            chosen_t = measured_hw['time'][index]
            chosen_t = chosen_t[peak == chosen_hhw]
            hhw[i] = chosen_hhw
            time_hhw[i] = chosen_t[0]
        else:
            hhw[i] = np.nan
            time_hhw[i] = np.nan

    while hhw.count(np.nan):
        hhw.remove(np.nan)
    while time_hhw.count(np.nan):
        time_hhw.remove(np.nan)

    if len(basic_hhw) > 3:
        hws_index = find_peaks(basic_hhw, distance=8)
        peak_hws = basic_hhw[hws_index]
        time_peak_hws = basic_time_hhw[hws_index]

        hws = np.empty(len(hws_index))
        time_hws = np.empty(len(hws_index))

        for i in range(len(hws_index)):
            index = np.where(
                time_hhw > time_peak_hws[i] - 2 &
                time_hhw < time_peak_hws[i] + 2
            )
            if index:
                peak = hhw[index]
                chosen_hws = max(peak)
                chosen_t = time_hhw[index]
                chosen_t = chosen_t[peak == chosen_hws]
                hws[i] = chosen_hws
                time_hws[i] = chosen_t[0]
            else:
                hws[i] = np.nan
                time_hws[i] = np.nan

        while hws.count(np.nan):
            hws.remove(np.nan)
        while time_hws.count(np.nan):
            time_hws.remove(np.nan)
    else:
        hws = np.nan
        time_hws = np.nan

    for i in range(0, len(basic_lw['hw'], 2)):
        lw = [basic_lw['lw'][i], basic_lw['lw'][i+1], basic_lw['lw'][i+2]]
        t = [basic_lw['time'][i], basic_lw['time'][i+1], basic_lw['time'][i+2]]
        if t[1] - t[0] < 16/24:
            chosen_llw = min(lw[0:1])
            chosen_t = t[lw == chosen_hhw]
            llw[i] = chosen_llw
            time_llw[i] = chosen_t[0]
        elif (
            t[1] - t[0] >= 16/24 and
            t[1] - t[0] < 32/24 and
            t[2] - t[1] >= 16/24 and
            t[2] - t[1] < 32/24
        ):
            chosen_llw = lw[0:1]
            chosen_t = t[0:1]
            llw[i:i+1] = chosen_llw
            time_llw[i:i+1] = chosen_t
        elif (
            t[1] - t[0] >= 16/24 and
            t[1] - t[0] < 32/24 and
            t[2] - t[1] < 16/24
        ):
            chosen_llw = max(lw)
            if chosen_llw == lw[2]:
                chosen_llw = lw[0:1]
                chosen_t = t[0:1]
                llw[i:i+1] = chosen_llw
                time_llw[i:i+1] = chosen_t
            else:
                llw[i] = hw[0]
                time_llw[i] = t[0]
        else:
            llw[i] = np.nan
            time_llw[i] = np.nan

    while llw.count(np.nan):
        llw.remove(np.nan)
    while time_llw.count(np.nan):
        time_llw.remove(np.nan)

    basic_llw = llw
    basic_time_llw = time_llw

    for i in range(len(llw)):
        index = np.where(
            measured_lw['time'] > basic_time_llw[i] - 8/24 &
            measured_lw['time'] > basic_time_llw[i] + 8/24
        )
        if index:
            peak = measured_lw['hw'][index]
            chosen_llw = max(peak)
            chosen_t = measured_lw['time'][index]
            chosen_t = chosen_t[peak == chosen_llw]
            llw[i] = chosen_llw
            time_llw[i] = chosen_t[0]
        else:
            llw[i] = np.nan
            time_llw[i] = np.nan

    while llw.count(np.nan):
        llw.remove(np.nan)
    while time_llw.count(np.nan):
        time_llw.remove(np.nan)

    if len(basic_llw) > 3:
        lws_index = find_peaks(basic_llw, distance=8)
        peak_lws = basic_llw[lws_index]
        time_peak_lws = basic_time_llw[lws_index]

        lws = np.empty(len(lws_index))
        time_lws = np.empty(len(lws_index))

        for i in range(len(lws_index)):
            index = np.where(
                time_llw > time_peak_lws[i] - 2 &
                time_llw < time_peak_lws[i] + 2
            )
            if index:
                peak = llw[index]
                chosen_lws = max(peak)
                chosen_t = time_llw[index]
                chosen_t = chosen_t[peak == chosen_lws]
                lws[i] = chosen_lws
                time_lws[i] = chosen_t[0]
            else:
                lws[i] = np.nan
                time_lws[i] = np.nan

        while lws.count(np.nan):
            lws.remove(np.nan)
        while time_lws.count(np.nan):
            time_lws.remove(np.nan)
    else:
        lws = np.nan
        time_lws = np.nan

    hhw = {'time': time_hhw, 'hw': hhw}
    llw = {'time': time_llw, 'hw': llw}
    hws = {'time': time_hws, 'hw': hws}
    lws = {'time': time_lws, 'hw': lws}

    return hhw, llw, hws, lws
