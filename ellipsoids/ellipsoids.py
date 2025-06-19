import numpy as np 
import cvxpy as cp



# Calculate Ellipsoid
def p_ball(xs, normp):
    xs = np.array(xs).T
    n = np.shape(xs)[0]
    m = np.shape(xs)[1]

    A = cp.Variable((n, n), symmetric=True)
    b = cp.Variable((n,1))

    # ------------------------------------------------------------------
    # objective --------------------------------------------------------
    n = A.shape[0]                       # matrix dimension
    objective = cp.Maximize(cp.log_det(A))

    # ------------------------------------------------------------------
    # constraints ------------------------------------------------------
    residual = A @ xs - b @ np.ones((1,m))
    col_norm_constraints = [
        cp.norm(residual[:, j], normp) <= 1
        for j in range(m)
    ]

    # element-wise bounds on A
    bounds = [A <= 1000 * np.ones((1,n)),   # upper element-wise bound
            A >= -1000 * np.ones((1,n))]  # lower element-wise bound

    constraints = col_norm_constraints + bounds 

    # ------------------------------------------------------------------
    # solve ------------------------------------------------------------

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)   

    print("Optimal value :", problem.value)
    print("A* =", A.value)
    print("b* =", b.value)
    return problem.value, A.value, b.value



