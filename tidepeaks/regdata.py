from datetime import datetime
import numpy as np
from scipy import interpolate


def regdata(time_raw: list[datetime], wl_raw: list[float], interval: float):
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
        time_fill[i] = time_raw_copy(index[i]) + interval/24
    time_raw_copy.append(time_fill)
    time_sorted_indices = np.argsort(time_raw_copy)
    time_sorted = time_raw_copy[time_sorted_indices]

    wl_fill = np.empty(len(index))
    wl_fill[:] = np.nan
    wl_raw_copy.append(wl_fill)
    wl_sorted = wl_raw_copy[time_sorted_indices]

    t1 = time_raw[0]
    t1.replace(minute=0, second=0, microsecond=0)
    time = np.arange(t1, time_raw[-1], interval/24)
    interp = interpolate.interp1d(time_sorted, wl_sorted)
    wl = interp(time)

    return time, wl
