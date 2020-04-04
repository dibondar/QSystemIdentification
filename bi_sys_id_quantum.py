import numpy as np
from scipy.linalg import logm, svd, lstsq, pinv, orthogonal_procrustes
from scipy.integrate import ode
from collections import namedtuple

def block_hankel(first_column, last_row):
    """
    Form block Hankel matrix from first_column and last_row.

    last_row[0] is ignored; the last row of the returned matrix is [first_column[-1], last_row[1:]].
    """
    blocks = [[]]

    # fillup with the first_column values
    for entry in first_column:
        for row in blocks:
            row.append(entry)
        blocks.append([])

    blocks.pop()

    # fillup with the last_row values
    lower_triangle = blocks[1:]

    for entry in last_row[1:]:
        for row in lower_triangle:
            row.append(entry)
        lower_triangle = lower_triangle[1:]

    return np.block(blocks)

def estimate_rank(orig_responces, alpha, print_sigma=True):
    """
    Estimate the rank based on a series of $m$-dimensional outputs under $r$ control
    :param orig_responces: numpy.array of dimension (r, number of time steps).
    :param alpha: (int) the shape of the Hankel matrix
    :param print_sigma: boolean flag indicating whether to print the singular values

    :return: (int) indicating rank
    """
    Y1 = orig_responces[0].T

    _, Sigma1, _ = svd(block_hankel(Y1[1:(alpha + 1)], Y1[alpha:]), full_matrices=False)

    rank = np.argmin(np.abs(Sigma1 / Sigma1.max() - 1e-3))

    if print_sigma:
        print("Estimated rank ", rank)
        print(Sigma1)

    return rank, Sigma1

# return type for the following function
Reconstructed = namedtuple('Reconstructed', ['Ac', 'C', 'Nc', 'x0'])

def bi_sys_id_my_version(orig_responces:np.ndarray, alpha:int, dt:float, v, rank:int=None):
    """
    Perform the bilinear system identification from a series of $m$-dimensional outputs
    under $r$ control.

    :param orig_responces:  numpy.array of observable data such that
                            orig_responces.shape = (p, r, m, the length of time series),
                            p -- the maximum duration of "on" pulse,
                            r -- the number of controls,
                            m -- the number of output.

    :param alpha: (int) the shape of the Hankel matrix
    :param dt: (float) time increment
    :param v: (list) the control values used to generate orig_responces

    :param rank: (int) rank to be used

    :return: Reconstructed
    """
    ####################################################################################################################
    #
    # identify rank, C, and A_c
    #
    ####################################################################################################################

    # declare parameters
    p = orig_responces.shape[0]  # number of measurements
    r = len(v)  # number of controls
    assert r == orig_responces.shape[1], "number of controls does not coincide"
    m = orig_responces.shape[2]  # number of outputs

    Y1 = orig_responces[0].T

    U1, Sigma1, V1_T = svd(block_hankel(Y1[1:(alpha + 1)], Y1[alpha:]), full_matrices=False)

    sqrt_sigma1 = np.sqrt(Sigma1)

    # I found the internally balanced realization to work the best in quantum case
    U1 *= sqrt_sigma1[None, ...]
    V1_T *= sqrt_sigma1[..., None]

    # estimate the rank if it is not given
    rank = rank if rank else np.argmin(np.abs(Sigma1 / Sigma1.max() - 1e-3))

    C_reconstructed = U1[:m, :rank]

    U1_up = U1[:-m, ]
    U1_down = U1[m:, ]

    Ac_reconstructed = (logm(
        # orthogonal_procrustes(U1_up, U1_down)[0][:rank, :rank]
        lstsq(U1_up, U1_down)[0][:rank, :rank]
    ) / dt)

    # Calculate B1_bar as save it into the B_bar list
    B_bar = [
        V1_T[:rank, :r]
    ]

    # truncate the rank of matrix
    U1 = U1[:, :rank]

    ####################################################################################################################
    #
    # Identify Nc[i] and the initial condition x0
    #
    ####################################################################################################################

    pinv_U1 = pinv(U1)

    B_bar.extend(
        pinv_U1 @ orig_responces[k - 1].T.reshape(-1, r)[(m * k):(m * k + m * alpha)] for k in range(2, p + 1)
    )

    C = [
        c.reshape(p, rank).T for c in np.hsplit(np.vstack(B_bar), 1)
    ]

    C_right = [c[:, 1:] for c in C]
    C_left = [c[:, :-1] for c in C]

    A_bar = [
        lstsq(left.T, right.T)[0].T for left, right in zip(C_left, C_right)
        #orthogonal_procrustes(left.T, right.T)[0].T for left, right in zip(C_left, C_right)
    ]

    Nc_reconstructed = [
        (logm(a) / dt - Ac_reconstructed) / vi for a, vi in zip(A_bar, v)
    ]

    # get the initial state
    lhs_x0 = []

    for a in A_bar:
        pow_a = a

        for _ in range(p):
            lhs_x0.append(pow_a)
            pow_a = pow_a @ a

    rhs_x0 = [c.T.reshape(-1, 1) for c in C]

    x0_reconstructed = lstsq(np.vstack(lhs_x0), np.vstack(rhs_x0), overwrite_a=True, overwrite_b=True)[0]

    return Reconstructed(
        Ac = Ac_reconstructed,
        C = C_reconstructed,
        Nc = Nc_reconstructed,
        x0 = x0_reconstructed,
    )

def get_response(model:Reconstructed, E, times:np.ndarray):
    """
    Calculate the response from the pulse E. We assume SINGLE valued outputs and SINGLE control (i.e., m=1 and r=1).
    :param model: the model represented by the data type of Reconstructed
    :param E: the real functions of time representing the filed
    :param times: (np.array) the time integration
    :return: np.array
    """

    # decide on the kind of integrator needed
    integrator_kind = 'zvode' if any(_.dtype == np.complex for _ in (model.x0, model.Ac, model.Nc[0])) else 'vode'

    solver = ode(
        lambda t, x, Ac, Nc, E: (Ac + Nc * E(t)) @ x
    ).set_integrator(integrator_kind)

    solver.set_initial_value(model.x0.reshape(-1), times[0]).set_f_params(model.Ac, model.Nc[0], E)

    # save the trajectory
    x = [solver.y]
    x.extend(
        solver.integrate(t) for t in times[1:]
    )
    #return np.array(x) @ model.C[0]
    return model.C @ np.array(x).T

def get_training_responses(model:Reconstructed, times:np.ndarray, p:int, u:float):
    """
    Recover responses that were used as input to reconstruct the model.
    We assume $m$-dimensional outputs and SINGLE control (i.e., r=1).
    :param model: the model represented by the data type of Reconstructed
    :param times: (np.array) the time integration
    :param p: (int) maximum number of time steps to keep the field on
    :param u: (float) the on value for the field
    :return: np.ndarray
    """

    # save responses
    responses = []

    # decide on the kind of integrator needed
    integrator_kind = 'zvode' if any(_.dtype == np.complex for _ in (model.x0, model.Ac, model.Nc[0])) else 'vode'

    for n in range(2, p + 2):

        ################################################################################################################
        #
        # the input is "on"
        #
        ################################################################################################################

        solver = ode(
            lambda t, x, Ac, Nc, u: (Ac + Nc * u) @ x
        ).set_integrator(integrator_kind)

        solver.set_initial_value(model.x0.reshape(-1), times[0]).set_f_params(model.Ac, model.Nc[0], u)

        # save the trajectory
        x = [solver.y]
        x.extend(
            solver.integrate(t) for t in times[1:n]
        )

        ################################################################################################################
        #
        # the input is "off"
        #
        ################################################################################################################

        solver = ode(
            lambda t, x, Ac: Ac @ x
        ).set_integrator(integrator_kind)

        solver.set_initial_value(x[-1], times[n - 1]).set_f_params(model.Ac)

        x.extend(
            solver.integrate(t) for t in times[n:]
        )

        ################################################################################################################
        #
        # calculate the output
        #
        ################################################################################################################

        responses.append(
            model.C @ np.array(x).T
        )

    return np.array(responses)