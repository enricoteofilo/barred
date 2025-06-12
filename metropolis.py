import os
import warnings
from tqdm import tqdm
import numpy as np

# METROPOLIS ALGORITHM PSEUDO-CODE
#loop
#    a is the present state of the Markov chain
#    select b with probability Aba = Aab
#    draw a random number in [0, 1) with uniform pdf
#    if r ≤ F (πb /πa ) then
#        the next state of the Markov chain is b
#    else
#        the next state of the Markov chain is a
#    end if
#end loop

#loop
#    xk is the present state of the Markov chain
#    select x̄ ∈ (xk − δ, xk + δ) with uniform pdf
#    select r ∈ [0, 1) with uniform pdf
#    if r ≤ min[1, f (x̄)/f (xk )] then
#    xk+1 = x̄
#    else
#    xk+1 = xk
#    end if
#end loop

def log_gaussian(x, mean, sigma):
    return -0.5*((x-mean)/sigma)**2 -0.5*np.log(2*np.pi) - np.log(sigma)

def log_multivariate_gaussian(x, mean, cov):
    return -0.5*np.matmul((x-mean),np.matmul(np.linalg.inv(cov),x-mean)) -0.5*len(x)*(2*np.pi) - 0.5*np.log(np.abs(np.linalg.det(cov)))

def gaussian_proposal(state, rng):
    sigma_proposal = 2.0
    if np.isscalar(state):
        return rng.normal(state,sigma_proposal)
    else:
        cov_matrix = np.diag((sigma_proposal**2)*np.ones_like(state))
        return rng.multivariate_normal(state,cov_matrix)    

def metropolis(log_target: callable, proposal: callable, rng, Nsamples, initial_state=None, chain=None):
    print("Initial state:", initial_state)
    if chain == None:
        chain = np.array([initial_state])
    else:
        init = initial_state
        try: initial_state = chain[len(chain) - 1]
        except:
            warnings.warn("Chain is empty, please provide an initial state or a non-empty existing chain.")
            chain = np.array([init])
            
    for n_sample in tqdm(range(Nsamples), desc=f"Sampling {Nsamples} from the target distribution..."):
        proposed_state = proposal(chain[-1], rng)
        y = (log_target(proposed_state)-log_target(chain[-1]))
        if y >= 0:
            chain = np.append(chain, [proposed_state], axis=0)
        else:
            if rng.uniform() <= np.min([1,np.exp(y)]):
                chain = np.append(chain,  [proposed_state], axis=0)
            else:
                chain = np.append(chain, [chain[-1]], axis=0)
    return chain
                
if __name__ == "__main__":    
    rng = np.random.default_rng(42)
    
    mu = 10.0
    sigma = 5.0 
    def log_target_distribution(x):
        return log_multivariate_gaussian(x, np.array([mu]), np.asarray([[sigma**2]]))
    
    x0 = np.array([0.0])
    N_samples = 10000
    
    print("x_0 is:",x0)
    
    chain = metropolis(log_target_distribution, gaussian_proposal, rng, Nsamples=N_samples, initial_state=x0)
    print(chain.shape)
    print(chain)
    
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(chain)),chain[:,0], label='Metropolis Chain')
    plt.axhline(10.0, color='red', linestyle='--', label='Target Mean')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Value')
    plt.title('Metropolis Sampling')
    plt.legend()
    plt.show()
    
    from scipy.signal import correlate
    autocorr = correlate(chain[:,0], chain[:,0], mode='full')/np.var(chain[:,0])
    plt.plot(autocorr, label='Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of Metropolis Chain')
    plt.legend()
    plt.show()
    print(f"Mean of the chain: {np.mean(chain)}")
    print(f"Standard deviation of the chain: {np.std(chain)}")
    
    plt.hist(chain[:,0], bins=50, density=True, alpha=0.5, label='Metropolis Samples')
    x = np.linspace(-20, 30, 1000)
    plt.plot(x, np.exp(log_gaussian(x,mu,sigma)), label='Target Distribution', color='red')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram of Metropolis Samples vs Target Distribution')
    plt.legend()
    plt.show()
    
        
        
        
        
    
        
