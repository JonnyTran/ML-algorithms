# Function to generate a bimodal distribution
def bimodal_random(n, mu_1=-1, sigma_1=0.05, mu_2=1, sigma_2=0.05):
    import numpy as np
    which_mode = np.random.binomial(1, 0.5, (n,))
    vals1 = np.random.randn(n) * sigma_1 + mu_1
    vals2 = np.random.randn(n) * sigma_2 + mu_2
    return vals1 * which_mode + vals2 * (1 - which_mode)
