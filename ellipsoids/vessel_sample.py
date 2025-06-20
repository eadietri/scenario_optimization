import os, sys

sys.path.insert(0, "../../src")

# Imports Gunnerus model
from mclsimpy.simulator import RVG_DP_6DOF

# Imports waves
from mclsimpy.waves import JONSWAP, WaveLoad

# Imports utilities
import numpy as np

from numpy.random import default_rng


def sample_vessel():
    dt = .1
    simtime = 25
    t = np.arange(0, simtime, dt)

    vessel = RVG_DP_6DOF(dt=dt, method='Euler')

    Uc = 0.0
    beta_c = np.pi/4

    eta = np.zeros((6, len(t)))
    nu = np.zeros((6, len(t)))
    tau_control = np.array([1000000, 0, 0, 0, 0, 2000000], dtype=float)

    hs = 1.0 # Significant wave height
    tp = 9.0 # Peak period
    gamma = 1.3 # Peak factor
    wp = 2*np.pi/tp # Peak frequency
    wmin = 0.5*wp
    wmax = 3.0*wp

    N = 50 # Number of wave components

    wave_freqs = np.linspace(wmin, wmax, N)

    jonswap = JONSWAP(wave_freqs)

    _, wave_spectrum = jonswap(hs=hs, tp=tp, gamma=gamma)

    dw = (wmax - wmin) / N
    wave_amps = np.sqrt(2 * wave_spectrum * dw)
    rand_phase = np.random.uniform(0, 2*np.pi, size=N)
    wave_angles = np.ones(N) * np.pi / 4

    waveload = WaveLoad(
        wave_amps=wave_amps,
        freqs=wave_freqs,
        eps=rand_phase,
        angles=wave_angles,
        config_file=vessel._config_file,
        interpolate=True,
        qtf_method="geo-mean",      # Use geometric mean to approximate the QTF matrices.
        deep_water=True,            # Assume deep water conditions.
    )

    rng = default_rng()
    ru = rng.uniform
    eta_0 = np.array([ru(5, 5.1), ru(0, 0), 
                   ru(0, 0),  ru(0, 0),
                   ru(0, 0.1), ru(10, 10.1)])
    eta_init = eta_0
    nu_init = np.zeros(6)

    vessel.set_eta(eta_init)
    vessel.set_nu(nu_init)
    reach = []

    for i in range(len(t)):
        tau_wave = waveload(t[i], vessel.get_eta())
        tau = tau_control + tau_wave
        eta[:, i] = vessel.get_eta()
        nu[:, i] = vessel.get_nu()
        vessel.integrate(Uc, beta_c, tau)
        
    for i in range(len(t)):
        reach.append([eta[0, i], eta[1, i], eta[2, i], eta[3, i], eta[4, i], eta[5, i]])
    
    reach=np.array(reach)
    
    return reach[-1]

def make_vessel_samples(ndata):
    return np.array([sample_vessel() for i in range(ndata)])


    