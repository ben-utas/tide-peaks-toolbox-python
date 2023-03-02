import numpy as np


def classpeaks(
    basic_hw: dict,
    basic_lw: dict,
    measured_hw: dict,
    measured_lw: dict
):
    time_hhw = np.empty(len(basic_hw['time']))
    hhw = np.empty(len(basic_hw['hw']))
    time_llw = np.empty(len(basic_lw['time']))
    llw = np.empty(len(basic_lw['lw']))

    for i in range(0, len(basic_hw['hw'], 2)):
        hw = [basic_hw['hw'][i], basic_hw['hw'][i+1], basic_hw['hw'][i+2]]
        t = [basic_hw['time'][i], basic_hw['time'][i+1], basic_hw['time'][i+2]]
        if t[1] - t[0] < 16/24:
            chosen_hhw = max(hw[0:1])
            t = t[hw == chosen_hhw]
            hhw[i] = chosen_hhw
            time_hhw[i] = t[0]
        elif (
            t[1] - t[0] >= 16/24 and
            t[1] - t[0] < 32/24 and
            t[2] - t[1] >= 16/24 and
            t[2] - t[1] < 32/24
        ):
            chosen_hhw = hw[0:1]
            t = t[0:1]
            hhw[i:i+1] = chosen_hhw
            time_hhw[i:i+1] = t
        
