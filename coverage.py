import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

###source : https://github.com/msesia/arc/blob/master/arc/coverage.py


def wsc(X, cover, delta=0.1, M=1000, random_state=2020, verbose=False):

    # Set up the default_rng generator using the extracted seed
    rng = np.random.default_rng(random_state)

    def wsc_v(X, cover, delta, v):
        n = np.shape(X)[0]
        #cover = np.array([ r[i] <= q[i] for i in range(n)])
        z = np.dot(X,v)

        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]


        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, cover, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, cover, delta, V[m])
        
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star




def wsc_unbiased(X, cover, delta=0.1, M=1000, test_size=0.8, random_state=2020, verbose=False):
    
    state = np.random.get_state()

    # Restore the original random state
    extracted_seed = state[1][0]
    
    def wsc_vab(X, cover, v, a, b):
        n = np.shape(X)[0]
        #cover = np.array([ r[i] <= q[i] for i in range(n)])
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, cover_train, cover_test = train_test_split(X, cover, test_size=test_size, random_state=extracted_seed)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, cover_train, delta=delta, M=M, random_state=extracted_seed, verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, cover_test, v_star, a_star, b_star)
    return coverage




