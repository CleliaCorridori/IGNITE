import numpy as np
from numba import njit, prange


@njit
def step_decay(epoch, start_lr = 1, drop = 0.5, edrop = 50):
    """
    -----------------------------------------------------------------
    Arguments:  - epoch, number of epochs, int
                - start_lr, initial learning rate, float
                - drop, percentage of the previous lr after drop,
                  float
                - edrop, number of epochs between each drop, int
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function implements a step-decaying learning rate for
    training.
    -----------------------------------------------------------------
    """
    lrate = start_lr*drop**np.floor((1+epoch)/edrop)
    return lrate


@njit
def cost_sign(x):
    """
    -----------------------------------------------------------------
    Arguments:  - x, numpy array of shape Nvar x Nsteps
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function is used to implement the L1 regularization in the
    likelihood maximization.
    -----------------------------------------------------------------
    """
    return 2*(x >= 0)*1 - 1


@njit
def find_theta_rec(h, J, x):
    """
    -----------------------------------------------------------------
    Arguments:  - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - x, numpy array of shape Nvar
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields, is
    used to evaluate the effective field on each spin which is used
    in the Glauber update formula.
    
    The function is used for the reconstruction step, where we need to
    use it on one-dimensional spins.
    -----------------------------------------------------------------
    """
    # np.fill_diagonal(J, 0) ##############
    
    return h + np.dot(J, x)
    

@njit
def find_theta(h, J, x):
    """
    -----------------------------------------------------------------
    Arguments:  - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - x, numpy array of shape Nvar x Nsteps
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields, is
    used to evaluate the effective field on each spin which is used
    in the Glauber update formula.
    -----------------------------------------------------------------
    """
    # np.fill_diagonal(J, 0) ##############
    
    return np.expand_dims(h, axis = 1) + np.dot(J, x)


@njit
def asynch_glauber_dynamics(h, J, spins, delta_t, gamma):
    """
    -----------------------------------------------------------------
    Arguments:  - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - x, numpy array of shape Nvar x Nsteps
                - delta_t, float
                - gamma, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields and a
    spin configuration at time t, generates the spin configuration at
    time t + delta_t.
    -----------------------------------------------------------------
    """

    theta = find_theta_rec(h, J, spins)
    p = gamma*delta_t/2
    r = np.random.rand(spins.shape[0])

    spins[r < p*(1-spins*np.tanh(theta))] = - spins[r < p*(1-spins*np.tanh(theta))]
    
    return spins


@njit
def generate_samples_asynch(h, J, delta_t, gamma, Nsteps, seed=0):
    """
    -----------------------------------------------------------------
    Arguments:  - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - delta_t, float
                - gamma, float
                - Nsteps, int
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields,
    generates Nsteps spins configuration, each separated by a delta_t
    interval.
    -----------------------------------------------------------------
    """
    Nspins = h.size
    np.random.seed(seed)
    # spins = np.ones(Nspins) # to check the initialization and the use of np.copy()
    spins = np.random.rand(Nspins)
    spins[spins < 1/2] = -1
    spins[spins > 1/2] = +1

    slist = np.empty((Nspins, Nsteps), dtype = np.float64)

    Ninitialise = Nsteps if Nsteps < 10e3 else 10e3
        
    for _ in range(Ninitialise):
        spins = asynch_glauber_dynamics(h, J, spins, delta_t, gamma)

    slist[:,0] = spins.copy()
    for idx in range(Nsteps-1):
        slist[:,idx+1] = asynch_glauber_dynamics(h, J, slist[:,idx].copy(), delta_t, gamma)

    return slist


@njit
def nb_cov(spins):
    """
    -----------------------------------------------------------------
    Arguments:  - spins, numpy array of shape Nvar x Nsteps
                - delta_t, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the covariance matrix averaging over time,
    using numba.
    -----------------------------------------------------------------
    """

    Nsteps = spins.shape[1]
    return np.dot(spins, spins.T)/Nsteps


@njit
def nb_dot_cov(spins, delta_t, cov):
    """
    -----------------------------------------------------------------
    Arguments:  - X, numpy array of shape Nvar x Nsteps
                - delta_t, float
                - cov, numpy array Nvar x Nvar
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the temporal derivative of the covariance
    matrix, which is used for the asynchronous reconstruction.
    -----------------------------------------------------------------
    """
    Nsteps = spins.shape[1]
    return 1/delta_t*(np.dot(spins[:,1:], spins[:,:-1].T)/Nsteps - cov)


@njit
def nb_dJ(cov, dot_cov, x, h, J, theta, gamma = 1):
    """
    -----------------------------------------------------------------
    Arguments:  - cov, numpy array of shape Nvar x Nvar
                - dot_cov, numpy array of shape Nvar x Nvar
                - x, numpy array of shape Nvar x Nsteps
                - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - theta, numpy array of shape Nvar x Nsteps
                - gamma, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields, is
    used to compute the gradients with respect to the couplings.
    -----------------------------------------------------------------
    """
    spins = x[:,:-1]
    dJ = np.empty(J.shape, dtype = np.float64)
    dJ = dot_cov/gamma + cov - np.dot(np.tanh(theta[:,:-1]), spins.T)/spins.shape[1]
    return dJ


@njit(parallel = True)
def nb_dh(x, h, J, theta):
    """
    -----------------------------------------------------------------
    Arguments:  - x, numpy array of shape Nvar x Nsteps
                - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - theta, numpy array of shape Nvar x Nsteps
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields, is
    used to compute the gradients with respect to the external fields.
    -----------------------------------------------------------------
    """
    spins = x[:,:-1]
    mean = np.empty(spins.shape[0], dtype = np.float64)
    vals = spins - np.tanh(theta[:,:-1])
    
    for idx in prange(spins.shape[0]):
        mean[idx] = np.mean(vals[idx])
    return mean


@njit
def momentum_update(dX, X, vX, eta_X, MOM, LAMBDA, L1):
    """
    -----------------------------------------------------------------
    Arguments:  - dX, numpy array
                - X, numpy array of the same shape as dX
                - vX, numpy array of the same shape as dX
                - eta_X, float
                - MOM, float
                - LAMBDA, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function takes the gradient and the past gradients to compute
    the gradient ascent step with momentum.
    -----------------------------------------------------------------
    """
    if L1:
        vX = MOM*vX + eta_X*(dX - LAMBDA*cost_sign(X))
    else:
        vX = MOM*vX + eta_X*(dX - LAMBDA*X)
    X += vX

    return vX, X


@njit
def momentum_reconstruct(spins, new_h, new_J, cov, dot_cov, delta_t, Nepochs = 500,
                         LAMBDA = 0.01, MOM = 0.95, L1 = False,
                         start_lr = 1, drop = 0.99, edrop = 20):
    """
    -----------------------------------------------------------------
    Arguments:  - spins, numpy array of shape Nvar x Nsteps
                - new_h, numpy array of shape Nvar
                - new_J, numpy array of shape Nvar x Nvar
                - delta_t, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function maximizes the likelihood using gradient ascent with
    momentum and L2 regularization.
    -----------------------------------------------------------------
    """    
    vJ = np.zeros(new_J.shape)
    vh = np.zeros(new_h.shape)
    
    eta_list = step_decay(np.arange(Nepochs), start_lr = start_lr, drop = drop, edrop = edrop)
    t = 0
    for eta in eta_list:
        theta = find_theta(new_h, new_J, spins)
        gh = nb_dh(spins, new_h, new_J, theta)
        gJ = nb_dJ(cov, dot_cov, spins, new_h, new_J, theta)
        
        old_J = new_J.copy()
        old_h = new_h.copy()
        
        vh, new_h = momentum_update(gh, new_h, vh, eta, MOM, LAMBDA, L1)
        vJ, new_J = momentum_update(gJ, new_J, vJ, eta, MOM, LAMBDA, L1)
        # np.fill_diagonal(new_J, 0)
        # np.fill_diagonal(gJ, 0)
        if t % 300 == 0:
            L = L_asynch(new_h, new_J, spins, cov, dot_cov, theta, delta_t)
            print(t, '\t', np.round(np.max(np.abs(gJ)), 6), '\t', np.round(np.max(np.abs(gh)), 6),
                  '\t', np.round(np.max(np.abs(new_J - old_J)), 4), '\t', np.round(np.max(np.abs(new_h - old_h)),4),
                  '\t', L)
        t += 1
    return new_h, new_J


@njit
def NADAM_update(m, v, dX, t, eta, X, LAMBDA, L1,
                beta_1 = 0.9, beta_2 = 0.999, eps = 10e-8):
    """
    -----------------------------------------------------------------
    Arguments:  - m, numpy array of the same shape as dX
                - v, numpy array of the same shape as dX
                - dX, numpy array of the same shape as dX
                - t, float
                - eta, float
                - X, numpy array of the same shape as dX
                - LAMBDA, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function computes the gradient ascent step with NADAM.
    -----------------------------------------------------------------
    """
    # m is the exp decaying average of past gradient 
    # v is the exp decaying average of past square gradients
    # t is the time
    m = beta_1*m + (1 - beta_1)*dX
    v = beta_2*v + (1 - beta_2)*np.power(dX, 2)
    m_hat = m/(1 - np.power(beta_1, t))
    v_hat = v/(1 - np.power(beta_2, t))
    
    if L1:
        u = X + eta*(1/(np.sqrt(v_hat) + eps)*(beta_1*m_hat + dX*(1-beta_1)/(1 - np.power(beta_1, t))) - LAMBDA*X)
    else:
        u = X + eta*(1/(np.sqrt(v_hat) + eps)*(beta_1*m_hat + dX*(1-beta_1)/(1 - np.power(beta_1, t))) - LAMBDA*cost_sign(X))

    return m, v, u


@njit
def NADAM_reconstruct(spins, new_h, new_J, cov, dot_cov, delta_t, Nepochs = 500, LAMBDA = 0.01,
                      L1 = False, start_lr = 1, drop = 0.99, edrop = 20):
    """
    -----------------------------------------------------------------
    Arguments:  - spins, numpy array of shape Nvar x Nsteps
                - new_h, numpy array of shape Nvar
                - new_J, numpy array of shape Nvar x Nvar
                - delta_t, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function maximizes the likelihood using gradient ascent with
    NADAM and regularization.
    -----------------------------------------------------------------
    """    
    mJ = np.zeros(new_J.shape)
    vJ = np.zeros(new_J.shape)
    
    mh = np.zeros(new_h.shape)
    vh = np.zeros(new_h.shape)
    
    # Early stopping ------------------------
    delta_L = np.zeros(25)
    L_list = []
    # ---------------------------------------
    
    eta_list = step_decay(np.arange(Nepochs), start_lr = start_lr, drop = drop, edrop = edrop)

    for t in np.arange(1, Nepochs+1):
        
        eta = eta_list[t-1]
        theta = find_theta(new_h, new_J, spins)
        gh = nb_dh(spins, new_h, new_J, theta)
        gJ = nb_dJ(cov, dot_cov, spins, new_h, new_J, theta)
        
        old_J = new_J.copy()
        old_h = new_h.copy()

        mh, vh, new_h = NADAM_update(mh, vh, gh, t, eta, new_h, LAMBDA, L1)
        mJ, vJ, new_J = NADAM_update(mJ, vJ, gJ, t, eta, new_J, LAMBDA, L1)
        # np.fill_diagonal(new_J, 0) #################
        # np.fill_diagonal(gJ, 0)
        if t % 300 == 0:
            L = L_asynch(new_h, new_J, spins, cov, dot_cov, theta, delta_t)
            print(t, '\t', np.round(np.max(np.abs(gJ)), 6), '\t', np.round(np.max(np.abs(gh)), 6),
                  '\t', np.round(np.max(np.abs(new_J - old_J)), 4), '\t', np.round(np.max(np.abs(new_h - old_h)),4),
                  '\t', np.round(L, 4))
            L_list.append(L)

        if len(L_list)>=3:
            delta_L= abs(L_list[-1] - L_list[-3])
            if delta_L<= 0.0001:
                print("early stopping implemented at epoch: ", t)
                break
        
    return new_h, new_J


@njit(parallel = True)
def L_asynch(h, J, x, covariance, dot_covariance, theta, delta_t, gamma = 1):
    """
    -----------------------------------------------------------------
    Arguments:  - h, numpy array of shape Nvar
                - J, numpy array of shape Nvar x Nvar
                - x, numpy array of shape Nvar x Nsteps
                - covariance, numpy array of shape Nvar x Nvar
                - dot_covariance, numpy array of shape Nvar x Nvar
                - theta, numpy array of shape Nvar x Nsteps
                - delta_t, float
                - gamma, float
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function, given the couplings and the external fields, is
    used to evaluate the likelihood over the data.
    -----------------------------------------------------------------
    """
    
    L = 0.
    
    for idx in prange(x.shape[0]):
        L += h[idx]*np.sum(x[idx]) - np.sum(np.log(2*np.cosh(theta[idx]))) \
             + np.sum(J[idx]*(dot_covariance[idx]/gamma + covariance[idx]))
        
    
    return L/x.shape[1]/x.shape[0]