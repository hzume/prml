import numpy as np
from matplotlib import pyplot as plt

def Gaussian(mu, sigma, x, epsilon=1e-5):
    K = len(mu)
    dim = len(x)
    C = np.array([(2*np.pi)**(dim/2) * np.abs(np.linalg.det(sigma[k]))**(1/2) for k in range(K)])
    
    inv_sigma = np.array([np.linalg.inv(sigma[k]) for k in range(K)])
    assert inv_sigma.shape == (K, dim, dim)
    assert (x-mu).shape == (K, dim)
    
    V = np.array([np.dot((x-mu)[k], np.dot(inv_sigma[k], (x-mu)[k])) for k in range(K)])
    assert V.shape == (K,)

    return  np.exp(-1/2*V) / C

def EM_mixed_gauss(K, X, max_iter=10):
    # init
    N = X.shape[0]
    dim = X.shape[1]
    mu = 4*np.random.rand(K, dim) - 2
    sigma = np.array([np.identity(dim) for _ in range(K)])
    pi = np.array([1/K]*K)
    # mu: [K, dim]
    # sigma: [K, dim, dim]
    # X: [N, dim]
    # pi: [K]
    # Gaussian(mu, sigma, x): [K]

    log_likelihood = 0
    for n in range(N):
        log_likelihood += np.log(np.sum(pi * Gaussian(mu, sigma, X[n])))
    print("0: ", log_likelihood)

    for i in range(max_iter):
        # E step
        gamma = np.array([pi * Gaussian(mu, sigma, X[n]) / np.sum(pi * Gaussian(mu, sigma, X[n])) for n in range(N)]).T
        assert gamma.shape == (K, N)
        
        
        # M step
        N_k = np.sum(gamma, axis=1)
        assert N_k.shape == (K, )
        

        mu_new = np.dot(gamma, X) / N_k[:, None]
        assert mu_new.shape == (K, dim)

        sigma_new = np.array([np.dot(gamma[k]*(X-mu_new[k]).T, X-mu_new[k]) for k in range(K)]) / N_k[:, None, None]
        assert sigma_new.shape == (K, dim, dim)

        pi_new = N_k / N
        assert pi_new.shape == (K,)

        mu = mu_new
        sigma = sigma_new
        pi = pi_new
        

        log_likelihood = 0
        for n in range(N):
            log_likelihood += np.log(np.sum(pi * Gaussian(mu, sigma, X[n])))
        print(f"{i+1}: ", log_likelihood)
    return gamma.T