import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from vessel_sample import make_vessel_samples
from ellipsoids import p_ball

# def f_full(a, b, c, d, e, f, A_full, b_full, normp=2):
#     # unwrap if any coordinate is still an array([x])
#     a, b, c, d, e, f = map(lambda v: float(v) if np.size(v) == 1 else v, (a, b, c, d, e, f))
#     vec = A_full @ np.array([a, b, c, d, e, f]) - b_full
#     return np.linalg.norm(vec, ord=normp)  
def ellipsoid_distance(x, A, b, p=2):
    """
    x : 1-D array-like, shape (n,)
    A : (n,n) SPD  matrix
    b : (n,)       vector
    """
    x = np.asarray(x, dtype=float).ravel()
    return np.linalg.norm(A @ x - b, ord=p)

def myminval(fn, x0):
    """
    MATLAB-like wrapper around Nelder–Mead.

    fn : Rᵏ→R   (k = len(x0))
    x0 : initial point  (scalar or 1-D array)
    """
    res = minimize(
        fn,
        x0=np.atleast_1d(x0),          # ensure array shape (k,)
        method="Nelder-Mead",
        options=dict(xatol=1e-4, fatol=1e-4, maxfev=200*len(np.atleast_1d(x0)),
                     disp=False))
    return res.fun        

def slice_xy_value(x, y, A, b, p, centre):
    """
    Minimise over the remaining 4 coords (z₁,z₂,z₃,z₄) while
    holding x,y fixed.
    """
    def objective(v):                         # v has length 4
        return ellipsoid_distance(
            np.concatenate(([x, y], v)), A, b, p)

    # good initial guess = centre’s coordinates for those dims
    return myminval(objective, centre[2:])  

def main():
    ndata = 100
    data = make_vessel_samples(ndata)
   
    ans = p_ball(data, 2)
    opt, A_full, b_full = ans
    b_full = b_full.flatten()
    xc = np.linalg.solve(A_full, b_full)

    # ------------------------------------------------------------------
    # build bounding box with 45 % margin ------------------------------
    # ------------------------------------------------------------------
    xlo = data.min(axis=0)    # [x_min, y_min, z_min]
    xhi = data.max(axis=0)    # [x_max, y_max, z_max]

    xmin = xlo[0] - 0.45 * (xhi[0] - xlo[0])
    xmax = xhi[0] + 0.45 * (xhi[0] - xlo[0])
    ymin = xlo[1] - 0.45 * (xhi[1] - xlo[1])
    ymax = xhi[1] + 0.45 * (xhi[1] - xlo[1])


 # ------------------------------------------------------------------
    # build grids -------------------------------------------------------
    # ------------------------------------------------------------------
    nmgrid = 100
    x_grid = np.linspace(xmin, xmax, nmgrid)
    y_grid = np.linspace(ymin, ymax, nmgrid)

    [X, Y] = np.meshgrid(x_grid, y_grid)

    Zxy_full = np.zeros((nmgrid, nmgrid))

    # ------------------------------------------------------------------
    # evaluate the objective on the grids ------------------------------
    # ------------------------------------------------------------------
    for i in range(nmgrid):
        for j in range(nmgrid):
            Zxy_full[i, j] = slice_xy_value(X[i, j], Y[i, j], A_full, b_full, 2, xc)


    # ------------------------------------------------------------------
    # plotting ----------------------------------------------------------
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    ax_xy = axes  # subplot(2,3,1)

    ax_xy.grid(True)

    # (x, y) plane
    ax_xy.plot(data[:, 0], data[:, 1], '.', color='gray', marker='.')
    ax_xy.contour(X, Y, Zxy_full, levels=[1], colors=['blue'], linewidths=1.5)
    ax_xy.set_xlabel(r'$x$', fontsize=12)
    ax_xy.set_ylabel(r'$y$', fontsize=12)
    ax_xy.set_title(r'Vessel', fontsize=13)
    ax_xy.set_xticks([-80, -75, -70, -65, -60, -55, -50, -45, -40])
    ax_xy.set_yticks([-150, -145, -140, -135, -130, -125, -120])

    plt.show()

if __name__ == '__main__':
    main()