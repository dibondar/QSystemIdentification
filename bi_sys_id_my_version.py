import numpy as np
from scipy.linalg import logm, hankel, svd, lstsq
from scipy.integrate import odeint
from collections import namedtuple

def estimate_rank(orig_responces, alpha, print_sigma=True):
    """
    Estimate the rank based on a series of SINGLE valued outputs and SINGLE control (i.e., m=1 and r=1)
    :param orig_responces: numpy.array of dimension (r, number of time steps).
    :param alpha: (int) the shape of the Hankel matrix
    :param print_sigma: boolean flag indicating whether to print the singular values

    :return: (int) indicating rank
    """
    Y1 = orig_responces[0]

    _, Sigma1, _ = svd(hankel(Y1[1:(alpha + 1)], Y1[alpha:]), full_matrices=False)

    rank = np.argmin(np.abs(Sigma1 / Sigma1.max() - 1e-3))

    if print_sigma:
        print("Estimated rank ", rank)
        print(Sigma1)

    return rank, Sigma1

# return type for the following function
Reconstructed = namedtuple('Reconstructed', ['Ac', 'C', 'Nc', 'x0'])

def bi_sys_id_my_version(orig_responces:np.ndarray, alpha:int, dt:float, v, rank:int=None):
    """
    Perform the bilinear system identification from a series of SINGLE valued outputs
    and SINGLE control (i.e., m=1 and r=1)

    :param orig_responces: numpy.array of dimension (r, number of time steps).
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
    m = 1  # number of outputs
    r = 1  # number of controls
    p = orig_responces.shape[0]  # number of measurements


    Y1 = orig_responces[0]

    U1, Sigma1, V1_T = svd(hankel(Y1[1:(alpha + 1)], Y1[alpha:]), full_matrices=False)

    # estimate the rank if it is not given
    rank = rank if rank else np.argmin(np.abs(Sigma1 / Sigma1.max() - 1e-3))

    C_reconstructed = U1[:m, :rank]

    U1_up = U1[:-m, ]
    U1_down = U1[m:, ]

    Ac_reconstructed = logm(
        lstsq(U1_up, U1_down)[0][:rank, :rank]
    ) / dt

    # Calculate B1_bar as save it into the B_bar list
    B_bar = [
        (Sigma1[..., None] * V1_T)[:rank, :r]
    ]

    # truncate the rank of matrix
    U1 = U1[:, :rank]
    ####################################################################################################################
    #
    # Identify Nc[i] and the initial condition x0
    #
    ####################################################################################################################

    B_bar.extend(
        U1.conj().T @ orig_responces[k - 1, k:(k + alpha)][..., None] for k in range(2, p + 1)
    )

    C = [
        c.reshape(p, rank).T for c in np.hsplit(np.vstack(B_bar), 1)
    ]

    C_right = [c[:, 1:] for c in C]
    C_left = [c[:, :-1] for c in C]

    A_bar = [
        lstsq(left.T, right.T)[0].T for left, right in zip(C_left, C_right)
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
    x = odeint(
        lambda x, t, Ac, Nc, E: (Ac + Nc * E(t)) @ x,
        model.x0.reshape(-1),
        times,
        args=(model.Ac, model.Nc[0], E)
    )

    return x @ model.C[0]

def get_training_responses(model:Reconstructed, times:np.ndarray, p:int, u:float):
    """
    Recover responses that were used as input to reconstruct the model.
    We assume SINGLE valued outputs and SINGLE control (i.e., m=1 and r=1).
    :param model: the model represented by the data type of Reconstructed
    :param times: (np.array) the time integration
    :param p: (int) maximum number of time steps to keep the field on
    :param u: (float) the on value for the field
    :return: np.ndarray
    """

    # save responses
    responses = []

    for n in range(2, p + 2):
        # the response is "on"
        x_on = odeint(
            lambda x, t, Ac, Nc, u: (Ac + Nc * u) @ x,
            model.x0.reshape(-1),
            times[:n],
            args=(model.Ac, model.Nc[0], u)
        )
        
        # the response is "off"
        x_off = odeint(
            lambda x, t, Ac: Ac @ x,
            x_on[-1],
            times[(n - 1):],
            args=(model.Ac,)
        )[1:]

        responses.append(
            np.vstack([x_on, x_off]) @ model.C[0]
        )

    return np.array(responses)
