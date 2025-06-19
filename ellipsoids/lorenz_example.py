import numpy as np 
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from ellipsoids import p_ball


plt.rcParams["text.usetex"]=True
plt.rcParams["font.family"]= "STIXGeneral"

def make_sample_n(sample_fn, parallel=True, pool=None):
    def sample_n(n, pool=pool):
        if parallel:
            if pool is None:
                print(r'Using {} CPUs'.format(cpu_count()))
                p = Pool(cpu_count())
            else:
                p = pool
            return np.array(list(p.map(sample_fn, np.arange(n))))
        else:
            return np.array([sample_fn() for i in range(n)])
    return sample_n

#Sampling scenarios for lorenz
def lorenz(x, t, sigma=10, beta=(8/3), rho=28):
    dxdt = [sigma*(x[1] - x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1] - beta*x[2]]
    return dxdt
def sample_lorenz(x=None):
    gamma = 0.5
    t = np.linspace(0, 100, 1001)
    rng = default_rng()
    y0 = np.array([1, 1, 1]) - gamma/2 + gamma * np.random.rand(3)
    sol = odeint(lorenz, y0, t)
    succ = sol[-1]
    return succ

sample_lorenz_n = make_sample_n(sample_lorenz)

def f_full(a, b, c, A_full, b_full, normp=2):
    # unwrap if any coordinate is still an array([x])
    a, b, c = map(lambda v: float(v) if np.size(v) == 1 else v, (a, b, c))
    vec = A_full @ np.array([a, b, c]) - b_full
    return np.linalg.norm(vec, ord=normp)  

def myminval(fn, x0):

    res = minimize(                     # Nelder–Mead in SciPy
        lambda z: fn(float(z[0])),      # unwrap array([z]) → scalar
        x0=np.atleast_1d(x0),           # x0 must be shape (1,)
        method="Nelder-Mead",
        options=dict(                   # MATLAB defaults
            xatol=1e-4,                 # TolX
            fatol=1e-4,                 # TolFun
            maxfev=200,                 # 200·n  (n = 1 here)
            disp=False))
    return res.fun   

def fpxy_full(x, y, A_full, b_full, normp, xc):
    return myminval(lambda z: f_full(x, y, xc[2], A_full, b_full, normp), 0)

def fpxz_full(x, z, A_full, b_full, normp, xc):
    return myminval(lambda y: f_full(x, y, z, A_full, b_full, normp), 0)



# Perform reachability experiment
def main():
    ndata = 1000
    sampler = sample_lorenz_n
    data = sampler(ndata)
    ans = p_ball(data, 2)
    opt, A_full, b_full = ans
    b_full = b_full.flatten()
    xc = np.linalg.solve(A_full, b_full)
    # ------------------------------------------------------------------
    # build bounding box with 35 % margin ------------------------------
    # ------------------------------------------------------------------
    xlo = data.min(axis=0)    # [x_min, y_min, z_min]
    xhi = data.max(axis=0)    # [x_max, y_max, z_max]

    xmin = xlo[0] - 0.35 * (xhi[0] - xlo[0])
    xmax = xhi[0] + 0.35 * (xhi[0] - xlo[0])
    ymin = xlo[1] - 0.35 * (xhi[1] - xlo[1])
    ymax = xhi[1] + 0.35 * (xhi[1] - xlo[1])

    zmin = xlo[2] - 0.35 * (xhi[2] - xlo[2])
    zmax = xhi[2] + 0.35 * (xhi[2] - xlo[2])

    # ------------------------------------------------------------------
    # build grids -------------------------------------------------------
    # ------------------------------------------------------------------
    nmgrid = 40
    x_grid = np.linspace(xmin, xmax, nmgrid)
    y_grid = np.linspace(ymin, ymax, nmgrid)
    z_grid = np.linspace(zmin, zmax, nmgrid)

    [X, Y] = np.meshgrid(x_grid, y_grid)
    [X, Z] = np.meshgrid(x_grid, z_grid)

    Zxy_full = np.zeros((nmgrid, nmgrid))
    Zxz_full = np.zeros((nmgrid, nmgrid))

    # ------------------------------------------------------------------
    # evaluate the objective on the grids ------------------------------
    # ------------------------------------------------------------------
    for i in range(nmgrid):
        for j in range(nmgrid):
            Zxy_full[i, j] = fpxy_full(X[i, j], Y[i, j], A_full, b_full, 2, xc)
            Zxz_full[i, j] = fpxz_full(X[i, j], Z[i, j], A_full, b_full, 2, xc)
    
    print(Zxy_full)

    # ------------------------------------------------------------------
    # plotting ----------------------------------------------------------
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(5, 5))
    ax_xy = axes[0]  # subplot(2,3,1)
    ax_xz = axes[1]

    ax_xy.grid(True)
    plt.setp(axes, xticks=[-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60], yticks=[-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60])

    # (x, y) plane
    ax_xy.plot(data[:, 0], data[:, 1], '.', color='gray', marker='.')
    ax_xy.contour(X, Y, Zxy_full, levels=[1], colors=['blue'], linewidths=1.5)
    ax_xy.set_xlabel(r'$x$', fontsize=12)
    ax_xy.set_ylabel(r'$y$', fontsize=12)
    ax_xy.set_title(r'Unconstrained', fontsize=13)

    ax_xz.grid(True)
    ax_xz.plot(data[:, 0], data[:, 2], '.', color='gray', marker='.')
    ax_xz.contour(X, Z, Zxz_full, levels=[1], colors=['blue'], linewidths=1.5)
    ax_xz.set_xlabel(r'$x$', fontsize=12)
    ax_xz.set_ylabel(r'$z$', fontsize=12)
    ax_xz.set_title(r'Unconstrained', fontsize=13)
    plt.show()


if __name__ == '__main__':
    main()
    
