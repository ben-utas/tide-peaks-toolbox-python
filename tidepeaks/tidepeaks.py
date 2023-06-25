from datetime import datetime

import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from utide import solve


class RegData:
    """
    Interpolates tide gauge time series to specified regular intervals.

    NOTES:
    - Duplicates are removed.
    - Gaps in the data greater than 3 hours are filled with NaNs.
    - The interpolated time series begins at the start of the first hour.
    - For further information refer to the Tide Peaks Toolbox User Manual.

    BEFORE RUNNING THIS FUNCTION:
    1. Create a new 'time_raw' variable datetime.
    2. Create a new 'wl_raw' variable in metres with the required vertical reference.
    3. Replace any error flag values in "wl_raw" with NaNs.
    4. Divide the time series into annual lots January to December.
        - Where available, include 6-12 hours before 1-Jan and after 31-Dec (ensures peaks occurring near New Years Eve are counted).

    Author: Karen Palmer, University of Tasmania
    Author: Ben Mildren, University of Tasmania
    Created: 22/02/2023

    Args:
        time_raw (list[datetime]): Input time vector in datetime format.
        wl_raw (list[float]): Input water levels in metres.
        interval (float): Specified output frequency in fraction of hours.

    Returns:
        time, wl: Regular interval time vector in datetime format and interpolated water levels in metres.
    """

    def __init__(self, time_raw: list[datetime], wl_raw: list[float], interval: float):
        # Remove duplicates
        index = np.where(np.diff(time_raw) == 0)
        time_raw.remove(index)
        wl_raw.remove(index)

        # Insert a single Nan into wl_raw vector 1 hour into gaps > 3 hours
        # When interpolation is executed wl values in the gaps will be NaNs
        index = np.where(np.diff(time_raw) > 3)
        time_raw_copy = time_raw
        wl_raw_copy = wl_raw
        time_fill = np.empty(len(index))
        time_fill[:] = np.nan

        # Insert extra time 1 interval into each gap
        for i in range(len(index)):
            time_fill[i] = time_raw_copy(index[i]) + interval / 24
        time_raw_copy.append(time_fill)
        time_sorted_indices = np.argsort(time_raw_copy)
        time_sorted = time_raw_copy[time_sorted_indices]

        wl_fill = np.empty(len(index))
        wl_fill[:] = np.nan
        wl_raw_copy.append(wl_fill)
        wl_sorted = wl_raw_copy[time_sorted_indices]

        t1 = time_raw[0]
        t1.replace(minute=0, second=0, microsecond=0)
        self.time = np.arange(t1, time_raw[-1], interval / 24)
        interp = interpolate.interp1d(time_sorted, wl_sorted)
        self.wl = interp(self.time)


class CoefDatums:
    """
    Takes the 'coef' structure output from UTide harmonic analysis and computes constituent-based tidal datums from standard formulae.

    NOTES:
    - Equations from ICSM Australian Tides Manual SP9 v.5.0, 2018 and Woodworth & Pugh, 2014. 'Sea Level Science'. Cambridge Press.
    - Expected constituents are computed from one calendar year Jan-Dec.
    - Additional datums can be computed from user specified equations.
    - Also computes the mean tide level and tidal form factor.
    - For further information refer to the Tide Peaks Toolbox User Manual.

    Author: Karen Palmer, University of Tasmania
    Author: Ben Mildren, University of Tasmania
    Created: 01/03/2023

    Args:
        coef (solve.Bunch): Tidal harmonic coefficient results structure from utide.

    Returns:
        hhwss, mhws, mhw, mlw, mlws, msl, mtl, mhhw, mllw, F (floats): high high water solstices springs, mean high water springs, mean higher high water, mean high water, mean sea level, mean tide level, mean low water, mean lower low water, mean low water springs, and form factor.
    """

    def __init__(self, coef: solve.Bunch):
        # Define variables retrieved through coef
        Z0 = coef.mean
        # Amplitude of each variable
        M2 = coef.A(np.isin(coef.name, "M2"))
        S2 = coef.A(np.isin(coef.name, "S2"))
        O1 = coef.A(np.isin(coef.name, "O1"))
        K1 = coef.A(np.isin(coef.name, "K1"))

        # Compute constituent tidal datums from standard formulae
        self.hhwss = Z0 + M2 + S2 + 1.4 * (K1 + O1)
        self.mhws = Z0 + (M2 + S2)
        self.mhw = Z0 + M2
        self.mlw = Z0 - M2
        self.mlws = Z0 - (M2 + S2)
        self.msl = Z0
        self.mtl = (self.mhw - self.mlw) / 2
        self.mhhw = Z0 + (M2 + K1 + O1)
        self.mllw = Z0 - (M2 + K1 + O1)
        self.form_factor = (K1 + O1) / (M2 + S2)


class Peaks:
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

    def __init__(
        self,
        time: list[datetime],
        wl_basic: list[float],
        wl_pred: list[float],
        wl: list[float],
    ):
        # Locate tidal maximas in basic tide model
        hw_index = find_peaks(wl_basic, distance=4 / 24, prominence=0.1)
        hw = wl_basic[hw_index]
        hw_time = time[hw_index]

        # Locate tidal minimas on inverted model
        lw_index = find_peaks(-wl_basic, distance=4 / 24, prominence=0.1)
        lw = wl_basic[hw_index]
        lw_time = time[hw_index]

        # Copy basic peaks to new variables
        bhw = hw
        bhw_time = hw_time
        blw = lw
        blw_time = lw_time

        # Output to dicts
        basic_hw = {"time": bhw_time, "hw": bhw}
        basic_lw = {"time": blw_time, "lw": blw}

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
                hw_time > bhw_time[i] - 4 / 24 & hw_time < bhw_time[i] + 4 / 24
            )
            if index:
                peaks = hw[index]
                high_water = max(peaks)
                t = hw_time[index]
                t = t[peaks == high_water]
                pred_hw["hw"][i] = high_water
                pred_hw["time"][i] = t[0]
            else:
                pred_hw["hw"][i] = np.nan
                pred_hw["time"][i] = np.nan
        for i in range(len(blw)):
            index = np.where(
                lw_time > blw_time[i] - 4 / 24 & lw_time < blw_time[i] + 4 / 24
            )
            if index:
                peaks = lw[index]
                low_water = min(peaks)
                t = lw_time[index]
                t = t[peaks == low_water]
                pred_lw["lw"][i] = low_water
                pred_lw["time"][i] = t[0]
            else:
                pred_lw["lw"][i] = np.nan
                pred_lw["time"][i] = np.nan

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
                hw_time > bhw_time[i] - 4 / 24 & hw_time < bhw_time[i] + 4 / 24
            )
            if index:
                peaks = hw[index]
                high_water = max(peaks)
                t = hw_time[index]
                t = t[peaks == high_water]
                obs_hw["hw"][i] = high_water
                obs_hw["time"][i] = t[0]
            else:
                obs_hw["hw"][i] = np.nan
                obs_hw["time"][i] = np.nan
        for i in range(len(blw)):
            index = np.where(
                lw_time > blw_time[i] - 4 / 24 & lw_time < blw_time[i] + 4 / 24
            )
            if index:
                peaks = lw[index]
                low_water = min(peaks)
                t = lw_time[index]
                t = t[peaks == low_water]
                obs_lw["lw"][i] = low_water
                obs_lw["time"][i] = t[0]
            else:
                obs_lw["lw"][i] = np.nan
                obs_lw["time"][i] = np.nan

        self.basic_hw = basic_hw
        self.basic_lw = basic_lw
        self.pred_hw = pred_hw
        self.pred_lw = pred_lw
        self.obs_hw = obs_hw
        self.obs_lw = obs_lw


class ClassPeaks:
    """
    Identifies high water peaks and low water troughs in tide data

    NOTES:
    - Classifies higher high water (HHW) and lower low water (LLW) from identified HW/LW
    - Values are given as time (datetime) and height (m)
    - Diurnal peaks are identified from the maximum peak and minimum trough in each sequence of two semidiurnal cycles, starting from the first crossing of Z0. Predicted HW/LW are identified from the 'Basic' tide fit using only the 8 most influential constituents (wl_basic) (this reduces misclassifications in complex tidal regimes)
    - For further information refer to the Tide Peaks Toolbox User Manual

    Args:
        basic_hw (dict): Dictionary of basic high water heights with 'time' keys storing datetime values and 'hw' keys storing floats.
        basic_lw (dict): Dictionary of basic high water heights with 'time' keys storing datetime values and 'lw' keys storing floats.
        measured_hw (dict): Dictionary of measured high water heights with 'time' keys storing datetime values and 'hw' keys storing floats.
        measured_lw (dict): Dictionary of measured low water heights with 'time' keys storing datetime values and 'lw' keys storing floats.

    Returns:
        hhw, llw, hws, lws (dicts): 'time' keys storing datetime values, and 'hhw', 'llw', 'hws', 'lws' keys respectively storing floats of higher high water, lower low water, higher semidiurnal water, and lower semidiurnal water.
    """

    def __init__(
        self, basic_hw: dict, basic_lw: dict, measured_hw: dict, measured_lw: dict
    ):
        time_hhw = np.empty(len(basic_hw["time"]))
        hhw = np.empty(len(basic_hw["hw"]))
        time_llw = np.empty(len(basic_lw["time"]))
        llw = np.empty(len(basic_lw["lw"]))

        for i in range(0, len(basic_hw["hw"], 2)):
            hw = [basic_hw["hw"][i], basic_hw["hw"][i + 1], basic_hw["hw"][i + 2]]
            t = [basic_hw["time"][i], basic_hw["time"][i + 1], basic_hw["time"][i + 2]]
            if t[1] - t[0] < 16 / 24:
                chosen_hhw = max(hw[0:1])
                chosen_t = t[hw == chosen_hhw]
                hhw[i] = chosen_hhw
                time_hhw[i] = chosen_t[0]
            elif (
                t[1] - t[0] >= 16 / 24
                and t[1] - t[0] < 32 / 24
                and t[2] - t[1] >= 16 / 24
                and t[2] - t[1] < 32 / 24
            ):
                chosen_hhw = hw[0:1]
                chosen_t = t[0:1]
                hhw[i : i + 1] = chosen_hhw
                time_hhw[i : i + 1] = chosen_t
            elif (
                t[1] - t[0] >= 16 / 24
                and t[1] - t[0] < 32 / 24
                and t[2] - t[1] < 16 / 24
            ):
                chosen_hhw = min(hw)
                if chosen_hhw == hw[2]:
                    chosen_hhw = hw[0:1]
                    chosen_t = t[0:1]
                    hhw[i : i + 1] = chosen_hhw
                    time_hhw[i : i + 1] = chosen_t
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
                measured_hw["time"]
                > basic_time_hhw[i] - 8 / 24 & measured_hw["time"]
                > basic_time_hhw[i] + 8 / 24
            )
            if index:
                peak = measured_hw["hw"][index]
                chosen_hhw = max(peak)
                chosen_t = measured_hw["time"][index]
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
                    time_hhw > time_peak_hws[i] - 2 & time_hhw < time_peak_hws[i] + 2
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

        for i in range(0, len(basic_lw["hw"], 2)):
            lw = [basic_lw["lw"][i], basic_lw["lw"][i + 1], basic_lw["lw"][i + 2]]
            t = [basic_lw["time"][i], basic_lw["time"][i + 1], basic_lw["time"][i + 2]]
            if t[1] - t[0] < 16 / 24:
                chosen_llw = min(lw[0:1])
                chosen_t = t[lw == chosen_hhw]
                llw[i] = chosen_llw
                time_llw[i] = chosen_t[0]
            elif (
                t[1] - t[0] >= 16 / 24
                and t[1] - t[0] < 32 / 24
                and t[2] - t[1] >= 16 / 24
                and t[2] - t[1] < 32 / 24
            ):
                chosen_llw = lw[0:1]
                chosen_t = t[0:1]
                llw[i : i + 1] = chosen_llw
                time_llw[i : i + 1] = chosen_t
            elif (
                t[1] - t[0] >= 16 / 24
                and t[1] - t[0] < 32 / 24
                and t[2] - t[1] < 16 / 24
            ):
                chosen_llw = max(lw)
                if chosen_llw == lw[2]:
                    chosen_llw = lw[0:1]
                    chosen_t = t[0:1]
                    llw[i : i + 1] = chosen_llw
                    time_llw[i : i + 1] = chosen_t
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
                measured_lw["time"]
                > basic_time_llw[i] - 8 / 24 & measured_lw["time"]
                > basic_time_llw[i] + 8 / 24
            )
            if index:
                peak = measured_lw["hw"][index]
                chosen_llw = max(peak)
                chosen_t = measured_lw["time"][index]
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
                    time_llw > time_peak_lws[i] - 2 & time_llw < time_peak_lws[i] + 2
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

        self.hhw = {"time": time_hhw, "hw": hhw}
        self.llw = {"time": time_llw, "lw": llw}
        self.hws = {"time": time_hws, "hw": hws}
        self.lws = {"time": time_lws, "lw": lws}

class PeakDatums:
    """
    Takes a series of identified HHW/HW/LW/LLW peaks/troughs and averages
    
    NOTES:
      - Expected time series has a period of one calendar year Jan-Dec *
      - Extracts the maximum (max_wl) and minimum (min_wl) levels in the period
      - Computes the Greater (greater_tidal_range) and Mean (mean_tidal_range) Tidal Range
      - Output datum variables are tables for each datum and respective deviation
      - For further information refer to the Tide Peaks Toolbox User Manual

    Inputs:
       time (list:datetime): Regular time list in datetime format.
       
   """    
    def __init__(
        self,
        time: list[datetime],
        wl: list[float],
        hws: dict,
        lws: dict,
        hhw: dict,
        llw: dict,
        hw: dict,
        lw: dict,
    ):
        max_wl = max(wl)
        max_time = time[wl.index(self.max_wl)][0]
        self.max_wl = {"time": max_time, "max": max_wl}
        
        min_wl = min(wl)
        min_time = time[wl.index(self.min_wl)][0]
        self.min_wl = {"time": min_time, "min": min_wl}
        
        msl = np.nanmean(wl)
        msl_mad = self.mad(wl)
        self.msl = {"mean": msl, "mad": msl_mad}
        
        mhws = np.nanmean(hws["hw"])
        mhws_mad = self.mad(hws["hw"])
        self.mhws = {"mean": mhws, "mad": mhws_mad}
        
        mhhw = np.nanmean(hhw["hw"])
        mhhw_mad = self.mad(hhw["hw"])
        self.mhhw = {"mean": mhhw, "mad": mhhw_mad}
        
        mhw = np.nanmean(hw["hw"])
        mhw_mad = self.mad(hw["hw"])
        self.mhw = {"mean": mhw, "mad": mhw_mad}
        
        mlw = np.nanmean(lw["hw"])
        mlw_mad = self.mad(lw["hw"])
        self.mlw = {"mean": mlw, "mad": mlw_mad}
        
        mllw = np.nanmean(llw["lw"])
        mllw_mad = self.mad(llw["lw"])
        self.mllw = {"mean": mllw, "mad": mllw_mad}
        
        mlws = np.nanmean(lws["lw"])
        mlws_mad = self.mad(lws["lw"])
        self.mlws = {"mean": mlws, "mad": mlws_mad}
        
        self.mtl = (self.mhhw["mean"] + self.mllw["mean"])/2
        
        self.greater_tidal_range = self.mhhw["mean"] - self.mllw["mean"]
        
        self.mean_tidal_range = mhw["mean"] - mlw["mean"]
    
    def mad(self, data: list[float], axis = None) -> float:
        """
        Computes the Mean Absolute Deviation of a list of values

        Inputs:
           data (list:float): List of values

        Outputs:
           mad (float): Mean Absolute Deviation
        """
        return np.nanmean(np.absolute(data - np.nanmean(data, axis)), axis)
