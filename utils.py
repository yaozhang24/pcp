import numpy as np
from scipy.stats.mstats import mquantiles
from sklearn.metrics import pairwise_distances, roc_auc_score, r2_score, pairwise_distances_argmin_min
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import DescrStatsW
from statistics import mode
from tqdm import tqdm
from sklearn.base import clone
import random


class PCP:
    """
        Posterior conformal prediction for generating prediction intervals

        Attributes:
            fold (int): Number of folds for cross-validation to choose the hyperparameters
            lambda_list (list): List of lambda values to choose
            grid (int): Number of grid points
            l (int): Smallest percentile for the grid points
            u (int): Largest percentile for the grid points
            m_min (int): Minimum number of samples used to generate the interval
        """

    def __init__(self, fold=20, grid=9, l=10, u=90, m_min=5):

        self.num_q = grid
        self.fold = fold
        self.l = l
        self.u = u
        self.lambda_list = [10 ** i for i in range(8, -8, -1)]
        self.lambda_ = []
        self.A_1_list = []
        self.A_1_inv_list = []
        self.X_tt_list = []
        self.H_list = []
        self.B_list = []
        self.test_list = []
        self.m = m_min
        self.E_Y = None
        self.R_Q = None
        self.n_c = None
        self.f = StandardScaler()
        self.state = np.random.get_state()

    def train(self, X, R, info=False):
        """
        Choose the hyperparameters in PCP

        Args:
            X (ndarray): Feature matrix used to pre-fit the predictive model
            R (ndarray): Response vector used to pre-fit the predictive model
            info (bool): If True, print the chosen hyperparameters. Default is False.
        """

        # Standardize the features
        self.f.fit(X)
        X_ = self.f.transform(X)
        X_ = np.concatenate([np.ones((len(X), 1)), X_], axis=1)

        # Calculate the grid points and generate the binary responses
        self.R_Q = np.percentile(R, np.linspace(self.l, self.u, self.num_q))
        Y_ = (R[:, None] <= self.R_Q).astype(float)

        # Choose the best lambda to estimate P{R < xi | X} for the grid points xi and return the estimators
        Pred = self._cross_validation(X_, Y_)

        # Determine the numer of clusters in our mixture model
        self._determine_number_of_clusters(Pred)

        # Construct the mixture model and return the membership probabilities and reconstruction r^2
        prob, _, r_squared = mixture(Pred, self.n_c, self.state)

        # Determine the sample size m to generate the PCP interval
        m_ = self._determine_precision(R.shape[0], prob)
        self.m = np.maximum(int(m_), self.m)

        # Print out information of our mixture model
        if info:
            print("r_square:", r_squared)
            print("number of components:", self.n_c)
            print("sample size m:", self.m)

    def _cross_validation(self, X_, Y_):

        kf = KFold(n_splits=self.fold, shuffle=False)
        Pred_list = []
        lts_list = []

        for lambda_ in self.lambda_list:
            Pred = np.zeros_like(Y_)
            lts_matrix = np.zeros((self.fold, self.num_q))

            for tt, (train_index, test_index) in enumerate(kf.split(X_)):
                X_t = X_[train_index]
                X_s = X_[test_index]
                Y_t = Y_[train_index]

                for t in range(self.num_q):
                    a_indices = Y_t[:, t] == 0
                    b_indices = Y_t[:, t] == 1

                    A_t = X_t[a_indices].T @ X_t[a_indices] + lambda_ * np.sum(a_indices) * np.identity(X_t.shape[1])
                    A_inv_t = np.linalg.inv(A_t) * np.sum(a_indices) / np.sum(b_indices)

                    h_t = np.sum(X_t[b_indices], axis=0)
                    b_t = np.dot(A_inv_t, h_t)
                    c_t = np.dot(X_s, b_t)
                    s_t = np.maximum(c_t, 0)
                    pred = s_t / (s_t + 1)
                    if is_all_zero_or_one(Y_[test_index, t]):
                        lts_matrix[tt, t] = -np.mean((pred >= 0.5) * Y_[test_index, t])
                    else:
                        lts_matrix[tt, t] = -roc_auc_score(Y_[test_index, t], pred)
                    Pred[test_index, t] = pred

            Pred_list.append(Pred)
            lts_list.append(np.mean(lts_matrix, axis=0)[:, None])

        L_mat = np.concatenate(lts_list, axis=1)
        min_idx = np.argmin(L_mat, axis=1)
        self.lambda_ = []

        Pred = np.zeros_like(Y_)
        for t in range(Pred.shape[1]):
            self.lambda_.append(self.lambda_list[min_idx[t]])
            Pred[:, t] = Pred_list[min_idx[t]][:, t]

        return Pred

    def _determine_number_of_clusters(self, Pred):
        """Determine the number of clusters in our mixture model."""
        n_clusters, r_squared_0, r_squared = 1, -1, 0

        while abs(r_squared - r_squared_0) > 0.05:
            n_clusters += 1
            r_squared_0 = r_squared

            prob, mu, r_squared = mixture(Pred, n_clusters, self.state)

        if abs(r_squared - r_squared_0) <= 0.025:
            self.n_c = n_clusters - 1
        else:
            self.n_c = n_clusters

    def _determine_precision(self, n, prob):

        """Determine the sample size m for generating the PCP interval."""
        m_max, m_min = 501, 4
        m_ = (m_max + m_min) / 2

        np.random.set_state(self.state)
        seeds = np.random.randint(1, 100000001, size=min(n, 1000))
        while abs(m_max - m_min) > 2:
            w_w_0 = np.zeros((min(n, 1000), min(n, 1000)))

            for j in range(min(n, 1000)):
                random.seed(seeds[j])
                idx = random.choices(population=range(self.n_c), weights=prob[j, :], k=int(m_))
                w_w_0[j, :] = np.sum(np.log(prob_clip(prob)[:min(n, 1000), idx]), axis=1)
                w_j = w_w_0[j, :]
                max_w = np.max(w_j)
                w_j = np.exp(w_j - max_w)
                w_j /= np.sum(w_j)
                w_w_0[j, :] = w_j

            n_hat = np.mean(1 / np.sum(w_w_0 ** 2, axis=1)) * min(max(n / 1000, 1), 1)
            n_hat_2 = np.mean(np.diagonal(w_w_0))

            if n_hat <= 100 and n_hat_2 >= 1 / 30:
                m_max = m_
            else:
                m_min = m_

            m_ = (m_max + m_min) / 2

        return m_

    def calibrate(self, X_val, R_val, X_test, R_test, alpha, return_pi=False, finite=False, max_iter=10, tol=0.005):
        """
         Generate prediction intervals for a set of testing points using a set of calibration (validation) points

         Args:
             max_iter: maximum number of iterations in alternating optimization
             tol: tolerance for terminating the alternating optimization algorithm
             X_val (ndarray): Validation feature matrix.
             R_val (ndarray): Validation response vector.
             X_test (ndarray): Test feature matrix.
             R_test (ndarray): Test response vector.
             alpha (float): Significance level.
             return_pi (bool): If True, return membership probabilities. Default is False.
             finite (bool): If True, store the infinite length as 2 x the largest prediction error. Default is False.

         Returns:
             list: List of quantiles for all the testing points
             list: List of coverage indicator for all the testing points
             list (optional): List of membership probabilities
         """

        # Standardize the features.
        X_val_ = self.f.transform(X_val)
        X_val_ = np.concatenate([np.ones((len(X_val_), 1)), X_val_], axis=1)

        X_test_ = self.f.transform(X_test)
        X_test_ = np.concatenate([np.ones((len(X_test_), 1)), X_test_], axis=1)

        # Calculate the grid points and generate the binary responses
        R_grids = np.concatenate([[0], self.R_Q, [np.inf]])
        Y_val_ = np.array(R_val[:, None] <= self.R_Q)

        # Prepare the matrix for cross-fitting the estimators of P{R < xi) for the grid points xi
        self._cross_fitting(X_val_, Y_val_)

        r_list = []
        coverage = []
        self.E_Y = np.zeros((X_val_.shape[0] + 1, self.num_q))

        np.random.set_state(self.state)
        seeds = np.random.randint(1, 100000001, size=np.shape(X_test_)[0])

        # Compute the interval for each testing point
        for j, x in tqdm(enumerate(X_test_)):
            # Loop over the grid points
            for t in np.arange(self.num_q, -1, -1):
                if t == self.num_q:
                    self.initial(x)
                else:
                    self.update(x, t)

                # Refit the mixture model
                prob, mu, _ = mixture(self.E_Y, self.n_c, self.state, max_iter=max_iter, tol=tol)
                random.seed(seeds[j])
                idx = random.choices(population=range(self.n_c), weights=prob[-1], k=self.m)
                s = np.sum(np.log(prob_clip(prob)[:, idx]), 1)
                max_s = np.max(s)
                weight = np.exp(s - max_s)
                weight /= np.sum(weight)
                wq = DescrStatsW(data=np.append(R_val, R_grids[t + 1]), weights=weight)
                r = wq.quantile(probs=1 - alpha, return_pandas=False)[0]

                # Stop if the weighted quantile is within the grid
                if r >= R_grids[t]:
                    coverage.append(R_test[j] <= r)
                    r_list.append(r)
                    break
                # Continue the algorithm if the quantile is below R_grids[t]

        # Replace the infinite interval lengths
        if finite:
            R_max = max(R_test)
            r_list = [R_max if x == np.inf else x for x in r_list]

        # Calculate the membership probabilities in the conditional guarantee
        if return_pi:
            pi_list = []

            np.random.set_state(self.state)
            seeds = np.random.randint(1, 100000001, size=np.shape(X_test_)[0])
            for j, x in tqdm(enumerate(X_test_)):

                k = np.sum(R_grids >= R_test[j]) - 1
                for t in np.arange(self.num_q, self.num_q - 1 - k, -1):
                    if t == self.num_q:
                        self.initial(x)
                    else:
                        self.update(x, t)

                prob, mu, _ = mixture(self.E_Y, self.n_c, self.state, max_iter=max_iter, tol=tol)
                random.seed(seeds[j])
                idx = random.choices(population=range(self.n_c), weights=prob[-1], k=self.m)
                pi_list.append([np.mean(np.array(idx) == i) for i in range(self.n_c)])
            return r_list, coverage, pi_list
        else:
            return r_list, coverage

    def _cross_fitting(self, X_val_, Y_val_):

        """Construct cross-fitted estimators for the probabilities P{R < xi | X} for the grid points xi"""
        kf = KFold(n_splits=self.fold, shuffle=False)
        for train_index, test_index in kf.split(X_val_):
            X_t = X_val_[train_index]
            X_tt = X_val_[test_index]
            Y_t = Y_val_[train_index]
            self.test_list.append(test_index)

            A_1, A_1_inv, H_, B_ = [], [], [], []
            for t in range(self.num_q):
                a_indices = Y_t[:, t] == 0
                b_indices = Y_t[:, t] == 1

                A_t = X_t[a_indices].T @ X_t[a_indices]
                A_t = A_t + self.lambda_[t] * np.sum(a_indices) * np.identity(X_t.shape[1])
                A_1.append(A_t + self.lambda_[t] * np.identity(X_t.shape[1]))

                h_t = np.sum(X_t[b_indices], axis=0)
                A_1_inv_t = np.linalg.inv(A_1[t]) * (np.sum(a_indices) + 1) / np.sum(b_indices)
                b_t_0 = np.dot(A_1_inv_t, h_t)

                A_1_inv.append(A_1_inv_t)
                H_.append(h_t)
                B_.append(b_t_0)

            self.A_1_list.append(A_1)
            self.A_1_inv_list.append(A_1_inv)
            self.X_tt_list.append(X_tt)
            self.H_list.append(H_)
            self.B_list.append(B_)

    def initial(self, x):
        """Initialize the prediction interval for y between the largest grid point and infinity"""
        for t in range(self.num_q):
            for tt in range(self.fold):
                test_index = self.test_list[tt]
                b_t = self.B_list[tt][t]
                Xx = self.X_tt_list[tt]

                if tt < self.fold - 1:
                    s_t = np.maximum(np.dot(Xx, b_t), 0)
                    self.E_Y[test_index, t] = s_t / (s_t + 1)
                else:
                    s_t = np.maximum(np.dot(np.vstack([Xx, x[None, :]]), b_t), 0)
                    self.E_Y[test_index, t] = s_t[:-1] / (s_t[:-1] + 1)
                    self.E_Y[-1, t] = s_t[-1] / (s_t[-1] + 1)

    def update(self, x, t):
        """Update the prediction interval for y between the t-th and (t+1)-th grid points"""

        for tt in range(self.fold):
            test_index = self.test_list[tt]
            h_t = self.H_list[tt][t]
            Xx = self.X_tt_list[tt]

            if tt < self.fold - 1:
                A_t = self.A_1_list[tt][t]
                A_inv_t = self.A_1_inv_list[tt][t]
                Ax = np.dot(A_t, x)
                Ax_inv = np.dot(A_inv_t, x)
                A_inv_t -= (np.dot(Ax_inv, Ax_inv.T) / (1 + np.dot(x, Ax)))

                b_t = np.dot(A_inv_t, h_t) + np.dot(A_inv_t, x)
                s_t = np.maximum(np.dot(Xx, b_t), 0)
                self.E_Y[test_index, t] = s_t / (s_t + 1)
            else:
                b_t = self.B_list[tt][t]
                s_t = np.maximum(np.dot(np.vstack([Xx, x[None, :]]), b_t), 0)
                self.E_Y[test_index, t] = s_t[:-1] / (s_t[:-1] + 1)
                self.E_Y[-1, t] = s_t[-1] / (s_t[-1] + 1)


def mirror_descent(Y, V, w, lr=0.1, max_iter=20):
    """
    Updates the membership probabilities by mirror descent

    """
    for t in range(max_iter):
        # Compute gradient
        grad = (w @ V - Y) @ V.T
        # Update weights using mirror descent step
        w = np.log(prob_clip(w)) - lr * grad / np.sqrt(lr * t + 1)
        # Normalize weights to get the softmax values
        w = softmax(w)

    return w


def update_v(Y, w):
    """
    Updates the cluster centers using the pseudo-inverse of the weight matrix.

    """
    return np.linalg.pinv(w.T @ w) @ w.T @ Y


def mixture(Pred, n_clusters, state, max_iter=20, tol=0.001):
    """
     Learn the membership probabilities by solving the reconstruction problem

     """

    # Set the columns with nan to zero
    # This occurs rarely in small datasets with no data within some grid points
    Y = replace_nan_columns_with_ones(Pred)

    # Standardize the prediction data
    Y = StandardScaler().fit_transform(Y)
    # Initialize cluster centers using k-means++
    V, w = kmeans_plus_plus(Y, n_clusters, state)

    # Compute initial R-squared value
    r_squared_old = r2_score(Y, w @ V)
    r_squared = 0
    for tt in range(max_iter):
        # Update weights using mirror descent
        w = mirror_descent(Y, V, w)
        # Update cluster centers

        V = update_v(Y, w)

        # Compute new R-squared value
        r_squared = r2_score(Y, w @ V)
        # Check for convergence
        if abs(r_squared - r_squared_old) <= tol:
            break
        r_squared_old = r_squared

    return w, V, r_squared


def kmeans_plus_plus_init(X, n_clusters):
    """
    Initializes cluster centers using the KMeans++ algorithm.
    """

    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Randomly choose the first center
    center_id = np.random.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of distances and calculate the first distances
    closest_dist_sq = np.full(n_samples, np.inf)
    distances = np.linalg.norm(X - centers[0], axis=1)
    closest_dist_sq = np.minimum(closest_dist_sq, distances ** 2)

    for i in range(1, n_clusters):
        # Choose the next center with a probability proportional to the squared distance
        rand_vals = np.random.random_sample(closest_dist_sq.shape) * closest_dist_sq
        candidate_id = np.argmax(rand_vals)
        centers[i] = X[candidate_id]

        # Update closest distances
        distances = np.linalg.norm(X - centers[i], axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, distances ** 2)

    return centers


def compute_inertia(X, centers):
    """
    Computes the inertia for the given data and centers.
    """

    # Find the closest center for each point and compute the distance
    closest_dist_sq, _ = pairwise_distances_argmin_min(X, centers)
    # Sum of squared distances of samples to their closest cluster center
    inertia = np.sum(closest_dist_sq ** 2)
    return inertia


def kmeans_plus_plus(X, n_clusters, state, n_init=20):
    """
    Applies the KMeans++ algorithm to find the best cluster centers.
    """

    best_inertia = np.inf
    best_centers = None
    np.random.set_state(state)

    for i in range(n_init):
        # Initialize centers using KMeans++ initialization
        init_centers = kmeans_plus_plus_init(X, n_clusters)
        # Compute inertia for the initialized centers
        inertia = compute_inertia(X, init_centers)

        # Keep track of the best centers with the lowest inertia
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = init_centers

    membership = softmax(-0.5 * pairwise_distances(X, best_centers) ** 2)

    return best_centers, membership


def prob_clip(prob, epsilon=1e-8):
    return np.maximum(prob, epsilon)


def prob_smoother(prob, epsilon=0.025):
    return (prob + epsilon) / (1 + np.shape(prob)[1] * epsilon)


def SCP(R_val, R_test, alpha=0.1, finite=False):
    """
     Computes prediction intervals of SCP
    """

    # Concatenate validation residuals with infinity to cover the upper bound in quantile calculation
    R_ = np.hstack((R_val, np.inf))

    # Compute uniform weights for the combined residuals
    w_window = np.ones_like(R_) / R_.shape[0]

    # Compute the weighted quantile for the given alpha
    wq = DescrStatsW(data=R_, weights=w_window)
    quantile_value = wq.quantile(probs=1 - alpha, return_pandas=False)[0]

    # Generate quantile predictions for the test residuals
    q = np.ones_like(R_test) * quantile_value

    # Compute coverage indicators (1 if the test residual is within the quantile, else 0)
    coverage = np.array(R_test <= q).tolist()

    if finite:
        R_max = max(R_test)
        q = [R_max if x == np.inf else x for x in q]
        return q, coverage
    else:
        return q.tolist(), coverage


def CC(Model, X_val, Y_val, X_test, Y_test, alpha=0.1, finite=False, exact=True):
    """
    Computes prediction intervals of SCP+CC
    """

    from conditionalconformal import CondConf

    # Add bias term (column of ones) to the validation and test feature matrices
    X_val = np.concatenate([np.ones((len(X_val), 1)), X_val], axis=1)
    X_test = np.concatenate([np.ones((len(X_test), 1)), X_test], axis=1)

    # Define score function (residuals) and inverse score functions for bounds
    score_fn = lambda x, y: y - Model.predict(x[:, 1:])
    score_inv_fn_ub = lambda s, x: [-np.inf, Model.predict(x.reshape(1, -1)[:, 1:]) + s]
    score_inv_fn_lb = lambda s, x: [Model.predict(x.reshape(1, -1)[:, 1:]) + s, np.inf]

    # Identity function for feature transformation
    phiFn = lambda x: x

    # Initialize conditional confidence interval calculator
    cond_conf = CondConf(score_fn, phiFn)
    cond_conf.setup_problem(X_val, Y_val)

    lbs = []  # List to store lower bounds
    ubs = []  # List to store upper bounds
    length = []  # List to store lengths of intervals
    coverage = []  # List to store coverage indicators

    for j in tqdm(range(X_test.shape[0])):
        x_t = X_test[j]

        # Predict lower bound for the j-th test sample
        lbs_t = cond_conf.predict(alpha / 2, x_t.reshape(1, -1), score_inv_fn_lb, exact=exact, randomize=True)[0][0]
        lbs.append(lbs_t)

        # Predict upper bound for the j-th test sample
        ubs_t = cond_conf.predict(1 - alpha / 2, x_t.reshape(1, -1), score_inv_fn_ub, exact=exact, randomize=True)[1][0]
        ubs.append(ubs_t)

        # Calculate length of the confidence interval
        length.append(abs(ubs_t - lbs_t))

        # Check if the true value is within the confidence interval
        coverage.append((Y_test[j] >= lbs_t) * (Y_test[j] <= ubs_t))

    if finite:
        R_test = abs(Y_test - Model.predict(X_test[:, 1:]))
        l_max = 2 * max(R_test)
        length = [l_max if x == np.inf else x for x in length]

    return lbs, ubs, length, coverage


def bandwidth_RLCP(X_data, n_=100):
    """
    Computes the bandwidth parameter for the RLCP algorithm using binary search.
    """

    # Get the number of samples (n) and the number of features (d)
    n, d = np.shape(X_data)

    # Initialize the minimum and maximum values for sigma
    min_sigma = 0
    max_sigma = 1000

    # Initial sigma as the midpoint of min_sigma and max_sigma
    sigma = (min_sigma + max_sigma) / 2

    # Perform binary search until the difference between max_sigma and min_sigma is less than 0.01
    while (max_sigma - min_sigma) > 0.001:
        # Compute pairwise squared distances
        dist = pairwise_distances(X_data, X_data) ** 2

        # Compute the affinity matrix using the Gaussian kernel
        w = np.exp(-sigma * dist)
        np.fill_diagonal(w, 0)  # Set diagonal to zero to ignore self-distances

        # Sum of affinities for each sample
        sum_w = np.sum(w, axis=1)

        # Compute the estimated mean local cluster size
        n_hat = n * np.mean((sum_w / (n - 1)) ** 2) / (((np.sum(w ** 2)) / (n * (n - 1))) + 1e-13)
        # Adjust sigma based on the comparison with the target value n_
        if n_hat < n_:
            max_sigma = sigma
        else:
            min_sigma = sigma

        # Update sigma as the midpoint of the new min_sigma and max_sigma
        sigma = (min_sigma + max_sigma) / 2

    return sigma


def RLCP(X_train, X_val, R_val, X_test, R_test, alpha=0.1, n=100, finite=False):
    """
    Computes prediction intervals of RLCP
    """

    # Standardize the training, validation, and test data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Estimate the optimal sigma using the RLCP bandwidth estimation function
    sigma_hat = bandwidth_RLCP(X_train, n_=n)

    q = []  # List to store the quantile predictions
    coverage = []  # List to store the coverage indicators
    n_test, d = np.shape(X_test)

    state = np.random.get_state()
    np.random.set_state(state)
    seeds = np.random.randint(1, 100000001, size=n_test)

    for j in range(n_test):
        random.seed(seeds[j])
        # Generate perturbed test sample
        X_test_tilde = X_test[j:j + 1, :] + np.array([random.gauss(0, 1 / np.sqrt(2 * sigma_hat)) for _ in range(d)])
        X_combined = np.concatenate([X_val, X_test[j:j + 1, :]], axis=0)

        # Compute weights based on the distance to the perturbed test sample
        w = np.exp(-sigma_hat * pairwise_distances(X_combined, X_test_tilde) ** 2)

        # Normalize weights
        w = w / np.sum(w)

        # Compute the weighted quantile for the residuals
        wq = DescrStatsW(data=np.hstack((R_val, np.inf)), weights=w)
        r = wq.quantile(probs=1 - alpha, return_pandas=False)[0]

        # Store the quantile prediction and the coverage indicator
        q.append(r)
        coverage.append(R_test[j] <= r)

    if finite:
        R_max = max(R_test)
        q = [R_max if x == np.inf else x for x in q]

    return q, coverage


class PCP_Group:

    def __init__(self, prob=None, group=None, m_min=20, m_max=500):
        """
         Initializes the PCP_Classification object.

         Parameters:
         - group: binary, indicate which group each sample belong to
         - prob: numpy array of shape (n_samples, n_classes), optional.
                 If provided, it contains the predictive probabilities for each sample.
         - m_min: int, optional, default=20.
                   Minimum number of samples used to generate the interval
         - m_max: int, optional, default=500.
                   Maximum number of samples used to generate the interval
         """

        if prob is None:
            self.m = m_min
        else:
            n, n_c = np.shape(prob)
            m_ = (m_max + m_min) / 2

            while abs(m_max - m_min) > 2:
                w_w_0 = np.zeros((min(n, 1000), min(n, 1000)))

                for j in range(min(n, 1000)):
                    idx = random.choices(population=range(n_c), weights=prob[j, :], k=int(m_))
                    w_w_0[j, :] = np.sum(np.log(prob_clip(prob)[:min(n, 1000), idx]), axis=1)
                    w_j = w_w_0[j, :]
                    max_w = np.max(w_j)
                    w_j = np.exp(w_j - max_w) * (group[:min(n, 1000)] == group[j])
                    w_j /= np.sum(w_j)
                    w_w_0[j, :] = w_j

                n_hat_2 = np.mean(np.diagonal(w_w_0))

                if n_hat_2 >= 1 / 30:
                    m_max = m_
                else:
                    m_min = m_

                m_ = (m_max + m_min) / 2

            self.m = int(m_)

    def calibrate(self, prob_v, prob_t, r_val, r_test, group_val, group_test, alpha, finite=False):
        """
        Calibrates the model using validation and test probability distributions.

        Parameters:
        - prob_v: numpy array of shape (n_val_samples, n_groups).
                  The predictive probabilities for the validation set.
        - prob_t: numpy array of shape (n_test_samples, n_groups).
                  The predictive probabilities for the test set.
        - r_val: numpy array of shape (n_val_samples,).
                 The residuals for the validation set.
        - r_test: numpy array of shape (n_test_samples,).
                  The residuals for the test set.
        - group_val: numpy array of shape (n_val_samples,).
                 The group indicators for the validation set.
        - group_test: numpy array of shape (n_test_samples,).
                 The group indicators for the test set.
        - alpha: 1- the target coverage rate
        - finite: If true, make the infinite interval infinite,
                 with length equal to 2*the largest testing residual

        Returns:
        - set_pcp: list of PCP quantiles
        - cover_pcp: list of coverage for each sample
        """

        prob_v = prob_smoother(prob_v)
        prob_t = prob_smoother(prob_t)

        n_val = np.shape(prob_v)[0]
        n_test = np.shape(prob_t)[0]

        w_w_0 = np.zeros((n_test, n_val + 1))

        coverage_pcp = []
        Q_pcp_list = []
        prob_list = []

        r_ = np.hstack((r_val, np.inf))
        r_max = max(r_test)

        for j in tqdm(range(n_test)):
            idx = np.random.choice(len(prob_t[j, :]), self.m, p=prob_t[j, :])
            w_w_0[j, -1] = np.sum(np.log(prob_clip(prob_t)[j, idx]))
            w_w_0[j, :-1] = np.sum(np.log(prob_clip(prob_v)[:, idx]), 1)
            w_j = w_w_0[j, :]
            max_w = max(w_j)
            w_j = np.exp(w_j - max_w) * (np.hstack([group_val, group_test[j]]) == group_test[j])
            w_j = w_j / np.sum(w_j)

            prob_list.append(np.sum(idx) / self.m)

            Q_pcp = DescrStatsW(r_, w_j).quantile(probs=1 - alpha, return_pandas=False)[0]

            if finite:
                if Q_pcp == np.inf:
                    Q_pcp = r_max

            Q_pcp_list.append(Q_pcp)
            coverage_pcp.append(r_test[j] <= Q_pcp)

        return Q_pcp_list, coverage_pcp, prob_list


class PCP_Classifcation:

    def __init__(self, prob=None, m_min=20, m_max=500):
        """
         Initializes the PCP_Classification object.

         Parameters:
         - prob: numpy array of shape (n_samples, n_classes), optional.
                 If provided, it contains the predictive probabilities for each sample.
         - m_min: int, optional, default=20.
                   Minimum number of samples used to generate the interval
         - m_max: int, optional, default=500.
                   Maximum number of samples used to generate the interval
         """

        if prob is None:
            self.m = m_min
        else:
            n, n_c = np.shape(prob)
            m_ = (m_max + m_min) / 2

            while abs(m_max - m_min) > 2:
                w_w_0 = np.zeros((min(n, 1000), min(n, 1000)))

                for j in range(min(n, 1000)):
                    idx = random.choices(population=range(n_c), weights=prob[j, :], k=int(m_))
                    w_w_0[j, :] = np.sum(np.log(prob_clip(prob)[:min(n, 1000), idx]), axis=1)
                    w_j = w_w_0[j, :]
                    max_w = np.max(w_j)
                    w_j = np.exp(w_j - max_w)
                    w_j /= np.sum(w_j)
                    w_w_0[j, :] = w_j

                n_hat_2 = np.mean(np.diagonal(w_w_0))

                if n_hat_2 >= 1 / 30:
                    m_max = m_
                else:
                    m_min = m_

                m_ = (m_max + m_min) / 2

            self.m = int(m_)

    def calibrate(self, prob_v, prob_t, y_val, y_test):
        """
        Calibrates the model using validation and test probability distributions.

        Parameters:
        - prob_v: numpy array of shape (n_val_samples, n_classes).
                  The predictive probabilities for the validation set.
        - prob_t: numpy array of shape (n_test_samples, n_classes).
                  The predictive probabilities for the test set.
        - y_val: numpy array of shape (n_val_samples,).
                 The true labels for the validation set.
        - y_test: numpy array of shape (n_test_samples,).
                  The true labels for the test set.

        Returns:
        - set_pcp: list of prediction sets
        - cover_pcp: list of booleans, indicating whether the true class is in the predicted set for each sample.
        - size_pcp: list of integers, representing the size of the prediction set for each sample.
        - p_pcp: list of floats, representing the top-class predictive probability for each sample
        """

        prob_v = prob_smoother(prob_v)
        prob_t = prob_smoother(prob_t)
        R_val = P_score(prob_v, y_val)

        cover_pcp = []
        set_pcp = []
        size_pcp = []
        p_pcp = []

        n_val = np.shape(prob_v)[0]
        n_test = np.shape(prob_t)[0]

        w_w_0 = np.zeros((n_test, n_val + 1))

        for j in tqdm(range(n_test)):
            idx = np.random.choice(len(prob_t[j, :]), self.m, p=prob_t[j, :])
            w_w_0[j, -1] = np.sum(np.log(prob_clip(prob_t)[j, idx]))
            w_w_0[j, :-1] = np.sum(np.log(prob_clip(prob_v)[:, idx]), 1)
            w_j = w_w_0[j, :]
            max_w = max(w_j)
            w_j = np.exp(w_j - max_w)
            w_j = w_j / np.sum(w_j)

            p = np.sum(idx == mode(idx)) / self.m

            cover_j, set_j = P_set(y_test[j], prob_t[j], R_val, p, w_j, allow_empty=True)
            set_pcp.append(set_j)
            cover_pcp.append(cover_j)
            size_pcp.append(len(set_j))
            p_pcp.append(p)

        return set_pcp, cover_pcp, size_pcp, p_pcp


class SCP_Classificaiton:
    def __init__(self, X_calib, Y_calib, black_box, alpha, random_state=1234, allow_empty=True):
        self.allow_empty = allow_empty

        # from msesia
        # https://github.com/msesia/arc/blob/master/arc/classification.py

        # Compute the size of the calibration set
        n2 = X_calib.shape[0]

        # Compute the predictive probabilities of the predictive model
        self.black_box = black_box
        p_hat_calib = self.black_box.predict_proba(X_calib)
        grey_box = ProbabilityAccumulator(p_hat_calib)

        # Compute the calibration level
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibration level
        self.alpha_calibrated = alpha - alpha_correction

    def predict(self, X, random_state=1234):
        # Compute the prediction set
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbabilityAccumulator(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat


class SplitConformal:
    def __init__(self, X_calib, Y_calib, black_box, alpha, random_state=1234, allow_empty=True):
        self.allow_empty = allow_empty

        # from msesia
        # https://github.com/msesia/arc/blob/master/arc/classification.py

        # Split data into training/calibration sets
        n2 = X_calib.shape[0]
        self.black_box = black_box

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib)
        grey_box = ProbabilityAccumulator(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

    def predict(self, X, random_state=1234):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbabilityAccumulator(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat


class ProbabilityAccumulator:

    # from msesia
    # https://github.com/msesia/arc/blob/master/arc/classification.py

    def __init__(self, prob):

        # Rank the predictive probabilities
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        self.Z = np.round(self.prob_sort.cumsum(axis=1), 9)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):

        # Compute the prediction set with randomization
        if alpha > 0:
            L = np.argmax(self.Z >= 1.0 - alpha, axis=1).flatten()
        else:
            L = (self.Z.shape[1] - 1) * np.ones((self.Z.shape[0],)).astype(int)
        if epsilon is not None:
            Z_excess = np.array([self.Z[i, L[i]] for i in range(self.n)]) - (1.0 - alpha)
            p_remove = Z_excess / np.array([self.prob_sort[i, L[i]] for i in range(self.n)])

            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1

        # Return prediction set
        S = [self.order[i, np.arange(0, L[i] + 1)] for i in range(self.n)]
        return S

    def calibrate_scores(self, Y, epsilon=None):

        # Compute the conformity scores
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([self.ranks[i, Y[i]] for i in range(n2)])
        prob_cum = np.array([self.Z[i, ranks[i]] for i in range(n2)])
        prob = np.array([self.prob_sort[i, ranks[i]] for i in range(n2)])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max


def P_set(y_test, p_test, R_val, target, weight=None, allow_empty=False):
    # from msesia
    # https://github.com/msesia/arc/blob/master/arc/classification.py

    # Rank the predictive probabilities
    prob_sort = -np.sort(-p_test)
    order = np.argsort(-p_test)

    R_ = np.hstack((R_val, 1.0))  # - prob_sort[0]

    if weight is None:
        beta_val = np.quantile(R_, target)
    else:
        beta_val = DescrStatsW(R_, weight).quantile(probs=target, return_pandas=False)[0]

    p_test_cum = np.round(prob_sort.cumsum(), 9)

    # Compute the prediction set
    if (p_test_cum < beta_val).all():
        set = []
    else:

        L = np.argmax(p_test_cum >= beta_val)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=1)

        p_test_cum_excess = p_test_cum[L] - beta_val

        p_remove = p_test_cum_excess / prob_sort[L]

        remove = epsilon <= p_remove

        if remove[0]:
            if not allow_empty:
                L = np.maximum(0, L - 1)
            else:
                L = L - 1

        set = order[np.arange(0, L + 1)]

    # Compute the coverage rate of the prediction set
    cover = np.sum(set == y_test)

    return cover, set


def P_score(prob, Y):
    # from msesia
    # https://github.com/msesia/arc/blob/master/arc/classification.py

    # Compute the conformity scores
    n, K = prob.shape
    order = np.argsort(-prob, axis=1)
    ranks = np.empty_like(order)
    for i in range(n):
        ranks[i, order[i]] = np.arange(len(order[i]))
    prob_sort = -np.sort(-prob, axis=1)
    epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
    prob_sum = np.round(prob_sort.cumsum(axis=1), 9)

    Y = np.atleast_1d(Y)
    ranks_y = np.array([ranks[i, Y[i]] for i in range(n)])
    prob_cum_y = np.array([prob_sum[i, ranks_y[i]] for i in range(n)])
    prob_y = np.array([prob_sort[i, ranks_y[i]] for i in range(n)])
    beta_max = prob_cum_y - np.multiply(prob_y, epsilon)

    beta_max = np.maximum(beta_max, 0)
    return beta_max


def train_val_test_split(X, Y, p, p2=None, return_index=False, random_state=None):
    """
    Splits the data into training, validation, and test sets and standardizes the features.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if p2 is None:
        p2 = p

    # Ensure Y is a 1D array
    Y = np.squeeze(Y)
    X = remove_constant_columns(X)
    # Number of samples
    n = X.shape[0]

    # Determine the size of the training and validation sets
    train_size = int(n * p)
    val_size = int(n * p2)
    # Calculate the end index of the validation set
    val_end = train_size + val_size

    # Shuffle the indices of the samples
    indices = np.random.permutation(n)  # Efficient way to shuffle indices

    # Split the indices into training, validation, and test sets
    idx_train = indices[:train_size]
    idx_val = indices[train_size:val_end]
    idx_test = indices[val_end:]

    # Split the data into training, validation, and test sets
    X_train_0, X_val_0, X_test_0 = X[idx_train], X[idx_val], X[idx_test]
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]

    # Standardize the training, validation, and test sets
    scaler = StandardScaler().fit(X_train_0)
    X_train = scaler.transform(X_train_0)
    X_val = scaler.transform(X_val_0)
    X_test = scaler.transform(X_test_0)

    if return_index:
        # Return the standardized data along with the original test set and test indices
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, X_test_0, idx_test
    else:
        # Return the standardized data along with the original test set
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, X_test_0


def softmax(x, axis=1):
    """
    Computes the softmax of each element along the specified axis of the input array.
    """

    # Subtract the maximum value along the specified axis to improve numerical stability
    max_x = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - max_x

    # Compute the exponentials of the shifted values
    exp_x_shifted = np.exp(x_shifted)

    # Sum of the exponentials along the specified axis
    sum_exp_x = np.sum(exp_x_shifted, axis=axis, keepdims=True)

    # Compute the softmax values by normalizing with the sum of exponentials
    softmax_values = exp_x_shifted / sum_exp_x

    return softmax_values


def is_all_zero_or_one(binary_vector):
    # Check if all elements are 0
    if all(x == 0 for x in binary_vector):
        return True
    # Check if all elements are 1
    elif all(x == 1 for x in binary_vector):
        return True
    else:
        return False


def remove_constant_columns(X):
    # Find indices of constant columns
    constant_columns = [i for i in range(X.shape[1]) if np.all(X[:, i] == X[0, i])]

    # Remove constant columns
    X_new = np.delete(X, constant_columns, axis=1)

    return X_new


def replace_nan_columns_with_ones(X):
    # Check if any value in each column is NaN
    nan_columns = np.isnan(X).any(axis=0)
    # Replace NaN columns with zeros
    X[:, nan_columns] = 1
    return X


def simulate_data(num_samples, setting):

    X = np.random.rand(num_samples, 1) * 8
    X2 = np.random.rand(num_samples, 5) * 8
    beta2 = np.random.randn(5)
    noise = np.random.normal(0, 1, num_samples)
    X_0 = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    X_ = np.concatenate([X_0, X * np.sin(X), X2], axis=1)
    beta = np.concatenate([np.array([-3, 1, -5]), beta2])

    if setting == 1:
        Y = np.dot(X_, beta) + 4 * noise * (1 + 0.5 * abs(X[:, 0] - 3) ** 2)
    if setting == 2:
        Y = np.dot(X_, beta) + 4 * noise * (1 + 3 * (X[:, 0] <= 5))
        # + np.dot(X2, beta2)
    X_out = np.concatenate([X, X2], axis=1)

    return X_out, Y


def cross_val_residuals(X_train, Y_train, model, n_splits=10, random_state=123456):
    """
    Perform cross-validation to compute residuals for hyperparameter selection.

    Parameters:
    - X_train: numpy array, training features.
    - Y_train: numpy array, training targets.
    - model: a scikit-learn compatible model with fit and predict methods.
    - n_splits: int, number of splits for cross-validation (default is 10).
    - random_state: int, random seed for reproducibility (default is 123456).

    Returns:
    - X_train_cv: numpy array, concatenated validation subsets.
    - R_train: numpy array, residuals from cross-validation.
    """
    R_train = []
    X_train_cv = []
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for train_index, val_index in tqdm(kf.split(X_train), total=n_splits, desc="Cross-validation"):
        model_clone = clone(model)  # Create a fresh copy of the model
        model_clone.fit(X_train[train_index], Y_train[train_index])
        Y_val_pred = model_clone.predict(X_train[val_index])
        X_train_cv.append(X_train[val_index])
        R_train.append(np.abs(Y_train[val_index] - Y_val_pred))

    return np.concatenate(X_train_cv), np.concatenate(R_train)


def cross_validation_classifier(X, y, model, n_splits=10, random_state=123456):
    """
    Perform cross-validation to predict probabilities for a target feature.

    Parameters:
    - X: numpy array, features.
    - y: numpy array, target values for cross-validation.
    - model: a scikit-learn compatible classifier with fit and predict_proba methods.
    - n_splits: int, number of splits for cross-validation (default is 10).
    - random_state: int, seed for reproducibility (default is 123456).

    Returns:
    - prob_cv: numpy array, predicted probabilities from cross-validation.
    """
    prob_cv = np.zeros((X.shape[0], 2))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X):
        model_clone = clone(model)  # Create a fresh copy of the model
        model_clone.fit(X[train_index], y[train_index])
        prob_cv[test_index] = model_clone.predict_proba(X[test_index])

    return prob_cv
