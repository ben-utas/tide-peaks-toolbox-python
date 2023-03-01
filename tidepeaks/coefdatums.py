import numpy as np
from utide import solve

def coefdatums(coef : solve.Bunch):
    """ Takes the 'coef' structure output from UTide harmonic analysis and computes constituent-based tidal datums from standard formulae.
    
    Author: Karen Palmer, University of Tasmania
    Author: Ben Mildren, University of Tasmania
    Created: 01/03/2023
    
    NOTES:
    - Equations from ICSM Australian Tides Manual SP9 v.5.0, 2018 and Woodworth & Pugh, 2014. 'Sea Level Science'. Cambridge Press.
    - Expected constituents are computed from one calendar year Jan-Dec *
    - Additional datums can be computed from user specified equations
    - Also computes the Mean Tide Level (MTL) and Tidal Form Factor (F)
    - For further information refer to the Tide Peaks Toolbox User Manual

    Args:
        coef (solve.Bunch): Tidal harmonic coefficient results structure from utide

    Returns:
        hhwss, mhws, mhw, mlw, mlws, msl, mtl, mhhw, mllw, F: high high water solstices springs, mean high water springs, mean higher high water, mean high water, mean sea level, mean tide level, mean low water, mean lower low water, mean low water springs, and form factor.
    """
    
    # Define variables retrieved through coef
    Z0 = coef.mean
    # Amplitude of each variable
    M2 = coef.A(np.isin(coef.name, 'M2'))
    S2 = coef.A(np.isin(coef.name, 'S2'))
    O1 = coef.A(np.isin(coef.name, 'O1'))
    K1 = coef.A(np.isin(coef.name, 'K1'))
    
    # Compute constituent tidal datums from standard formulae
    hhwss = Z0 + M2 + S2 + 1.4*(K1 + O1)
    mhws = Z0 + (M2 + S2)
    mhw = Z0 + M2
    mlw = Z0 - M2
    mlws = Z0 - (M2 + S2)
    msl = Z0
    mtl = (mhw-mlw)/2
    mhhw = Z0 + (M2 + K1 + O1)
    mllw = Z0 - (M2 + K1 + O1)
    # Calculate form factor
    F = (K1 + O1)/(M2 + S2)
    
    return hhwss, mhws, mhw, mlw, mlws, msl, mtl, mhhw, mllw, F