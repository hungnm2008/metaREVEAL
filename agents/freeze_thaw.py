from agents.base_agent import *
import time
import numpy as np
import scipy.stats

"""
This code is adapted from: https://github.com/jamesrobertlloyd/automl-phase-1
"""

def trunc_norm_mean_upper_tail(a, mean, std):
    alpha = (a - mean) / std
    num = scipy.stats.norm.pdf(alpha)
    den = (1 - scipy.stats.norm.cdf(alpha))
    if num == 0 or den == 0:
        # Numerical nasties
        if a < mean:
            return mean
        else:
            return a
    else:
        lambd = scipy.stats.norm.pdf(alpha) / (1 - scipy.stats.norm.cdf(alpha))
        return mean + std * lambd


def ft_K_t_t(t, t_star, scale, alpha, beta):
    """
    Exponential decay mixture kernel
    """
    # Check 1d
    # TODO - Abstract this checking behaviour - check pybo and gpy for inspiration
    t = np.array(t)
    t_star = np.array(t_star)
    assert t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1)
    assert t_star.ndim == 1 or (t_star.ndim == 2 and t_star.shape[1] == 1)
    # Create kernel
    K_t_t = np.zeros((len(t), len(t_star)))
    for i in range(len(t)):
        for j in range(len(t_star)):
            K_t_t[i, j] = scale * (beta ** alpha) / ((t[i] + t_star[j] + beta) ** alpha)
    return K_t_t


def ft_K_t_t_plus_noise(t, t_star, scale, alpha, beta, log_noise):
    """
    Ronseal - clearly this behaviour should be abstracted
    """
    # TODO - abstract kernel addition etc
    noise = np.exp(log_noise)
    K_t_t = ft_K_t_t(t, t_star, scale=scale, alpha=alpha, beta=beta)
    K_noise = cov_iid(t, t_star, scale=noise)
    return K_t_t + K_noise


def cov_iid(x, z=None, scale=1):
    """
    Identity kernel, scaled
    """
    if z is None:
        z = x
    # Check 1d
    x = np.array(x)
    z = np.array(z)
    assert x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)
    assert z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1)
    # Create kernel
    K = np.zeros((len(x), len(z)))
    if not np.all(x == z):
        # FIXME - Is this the correct behaviour?
        return K
    for i in range(min(len(x), len(z))):
        K[i, i] = scale
    return K


def cov_matern_5_2(x, z=None, scale=1, ell=1):
    """
    Identity kernel, scaled
    """
    if z is None:
        z = x
    # Check 1d
    x = np.array(x, ndmin=2)
    z = np.array(z, ndmin=2)
    if x.shape[1] > 1:
        x = x.T
    if z.shape[1] > 1:
        z = z.T
    assert (x.ndim == 2 and x.shape[1] == 1)
    assert (z.ndim == 2 and z.shape[1] == 1)
    # Create kernel
    x = x * np.sqrt(5) / ell
    z = z * np.sqrt(5) / ell
    sqdist = np.sum(x**2,1).reshape(-1,1) + np.sum(z**2,1) - 2*np.dot(x, z.T)
    K = sqdist
    f = lambda a: 1 + a * (1 + a / 3)
    m = lambda b: f(b) * np.exp(-b)
    for i in range(len(K)):
        for j in range(len(K[i])):
            K[i, j] = m(K[i, j])
    K *= scale
    return K


def slice_sample_bounded_max(N, burn, logdist, xx, widths, step_out, max_attempts, bounds):
    """
    Slice sampling with self.bounds and max iterations
    Iain Murray May 2004, tweaks June 2009, a diagnostic added Feb 2010
    See Pseudo-code in David MacKay's text book p375
    Modified by James Lloyd, May 2012 - max attempts
    Modified by James Lloyd, Jan 2015 - self.bounds
    Ported to python by James Lloyd, Feb 2015
    """
    xx = copy.deepcopy(xx)
    D = len(xx)
    samples = []
    if (not isinstance(widths, list)) or len(widths) == 1:
        widths = np.ones(D) * widths

    log_Px = logdist(xx)

    for ii in range(N + burn):
        log_uprime = np.log(random.random()) + log_Px
        for dd in random.sample(range(D), D):
            x_l = copy.deepcopy(xx)
            x_r = copy.deepcopy(xx)
            xprime = copy.deepcopy(xx)

            # Create a horizontal interval (x_l, x_r) enclosing xx
            rr = random.random()
            x_l[dd] = max(xx[dd] - rr*widths[dd], bounds[dd][0])
            x_r[dd] = min(xx[dd] + (1-rr)*widths[dd], bounds[dd][1])

            if step_out:
                while logdist(x_l) > log_uprime and x_l[dd] > bounds[dd][0]:

                    x_l[dd] = max(x_l[dd] - widths[dd], bounds[dd][0])
                while logdist(x_r) > log_uprime and x_r[dd] < bounds[dd][1]:
                    x_r[dd] = min(x_r[dd] + widths[dd], bounds[dd][1])

            # Propose xprimes and shrink interval until good one found
            zz = 0
            num_attempts = 0
            while True:
                zz += 1
                # print(x_l)
                xprime[dd] = random.random()*(x_r[dd] - x_l[dd]) + x_l[dd]
            
                log_Px = logdist(xx)
                if log_Px > log_uprime:
                    xx[dd] = xprime[dd]
                    break
                else:
                    # Shrink in
                    num_attempts += 1
                    if num_attempts >= max_attempts:
                        # print('Failed to find something')
                        break
                    elif xprime[dd] > xx[dd]:
                        x_r[dd] = xprime[dd]
                    elif xprime[dd] < xx[dd]:
                        x_l[dd] = xprime[dd]
                    else:
                        raise Exception('Slice sampling failed to find an acceptable point')
        # Record samples
        if ii >= burn:
            samples.append(copy.deepcopy(xx))
    return samples


# noinspection PyTypeChecker
def ft_ll(m, t, y, x, x_kernel, x_kernel_params, t_kernel, t_kernel_params):
    """
    Freeze thaw log likelihood
    """
    # Take copies of everything - this is a function
    m = copy.deepcopy(m)
    t = copy.deepcopy(t)
    y = copy.deepcopy(y)
    x = copy.deepcopy(x)

    K_x = x_kernel(x, x, **x_kernel_params)
    N = len(y)

    lambd = np.zeros((N, 1))
    gamma = np.zeros((N, 1))

    K_t = [None] * N

    for n in range(N):
        K_t[n] = t_kernel(t[n], t[n], **t_kernel_params)
        lambd[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], np.ones((len(t[n]), 1))))
        # Making sure y[n] is a column vector
        y[n] = np.array(y[n], ndmin=2)
        if y[n].shape[0] == 1:
            y[n] = y[n].T
        gamma[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], y[n] - m[n] * np.ones(y[n].shape)))

    Lambd = np.diag(lambd.ravel())

    ll = 0

    # Terms relating to individual curves
    for n in range(N):
        ll += - 0.5 * np.dot((y[n] - m[n] * np.ones(y[n].shape)).T,
                             np.linalg.solve(K_t[n], y[n] - m[n] * np.ones(y[n].shape)))
        ll += - 0.5 * np.log(np.linalg.det(K_t[n]))

    # Terms relating to K_x
    ll += + 0.5 * np.dot(gamma.T, np.linalg.solve(np.linalg.inv(K_x) + Lambd, gamma))
    ll += - 0.5 * np.log(np.linalg.det(np.linalg.inv(K_x) + Lambd))
    ll += - 0.5 * np.log(np.linalg.det(K_x))

    return ll


# noinspection PyTypeChecker
def ft_posterior(m, t, y, t_star, x, x_kernel, x_kernel_params, t_kernel, t_kernel_params):
    """
    Freeze thaw posterior (predictive)
    """
    # Take copies of everything - this is a function
    m = copy.deepcopy(m)
    t = copy.deepcopy(t)
    y = copy.deepcopy(y)
    t_star = copy.deepcopy(t_star)
    x = copy.deepcopy(x)

    K_x = x_kernel(x, x, **x_kernel_params)
    N = len(y)

    lambd = np.zeros((N, 1))
    gamma = np.zeros((N, 1))
    Omega = [None] * N

    K_t = [None] * N
    K_t_t_star = [None] * N

    y_mean = [None] * N

    for n in range(N):
        K_t[n] = t_kernel(t[n], t[n], **t_kernel_params)
        # TODO - Distinguish between the curve we are interested in and 'noise' with multiple kernels
        K_t_t_star[n] = t_kernel(t[n], t_star[n], **t_kernel_params)
        lambd[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], np.ones((len(t[n]), 1))))
        # Making sure y[n] is a column vector
        y[n] = np.array(y[n], ndmin=2)
        if y[n].shape[0] == 1:
            y[n] = y[n].T
        gamma[n] = np.dot(np.ones((1, len(t[n]))), np.linalg.solve(K_t[n], y[n] - m[n] * np.ones(y[n].shape)))
        Omega[n] = np.ones((len(t_star[n]), 1)) - np.dot(K_t_t_star[n].T,
                                                         np.linalg.solve(K_t[n], np.ones(y[n].shape)))

    Lambda_inv = np.diag(1 / lambd.ravel())
    C = K_x - np.dot(K_x, np.linalg.solve(K_x + Lambda_inv, K_x))
    mu = m + np.dot(C, gamma)
    for n in range(N):
        y_mean[n] = np.dot(K_t_t_star[n].T, np.linalg.solve(K_t[n], y[n])) + Omega[n] * mu[n]

    K_t_star_t_star = [None] * N
    y_var = [None] * N

    for n in range(N):
        K_t_star_t_star[n] = t_kernel(t_star[n], t_star[n], **t_kernel_params)
        y_var[n] = K_t_star_t_star[n] - \
                   np.dot(K_t_t_star[n].T,
                          np.linalg.solve(K_t[n], K_t_t_star[n])) + \
                   C[n, n] * np.dot(Omega[n], Omega[n].T)

    return y_mean, y_var

def colorbrew(i):
    """
    Nice colors taken from http://colorbrewer2.org/ by David Duvenaud March 2012
    """
    rgbs = [(228,  26,  28),
            (55, 126, 184),
            (77, 175,  74),
            (152,  78, 163),
            (255, 127, 000),
            (255, 255, 51),
            (166,  86, 40),
            (247, 129, 191),
            (153, 153, 153),
            (000, 000, 000)]
    # Convert to [0, 1] range
    rgbs = [(r / 255, g / 255, b / 255) for (r, g, b) in rgbs]
    # Return color corresponding to index - wrapping round
    return rgbs[i % len(rgbs)]

class Freeze_Thaw(Base_Agent):
    """
    Model the Freeze_Thaw agent
    """
    def __init__(self, agent_name, env):
        super().__init__(agent_name, env)
        self.initialize()

    def initialize(self):
        """
        Initialization
        """
        self.start_thinking = False
        self.compute_quantum = self.env.time_for_each_action
#         self.all_pickles = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
        self.all_pickles = [[i] for i in range(self.env.nA)]
        self.predict_counter = 0
        self.msl=[(i+1) for i in range(self.env.nA)]

        self.x = []
        for i in self.msl:
            self.x.append([np.log(i)])

        self.scores = []
        self.times = []
        self.prediction_times = []
        self.prediction_files = []
        self.models_to_be_run = [i for i in range(self.env.nA)]
        self.alpha = []
        self.beta = []
        self.scale = []
        self.log_noise = []
        self.x_scale = []
        self.x_ell = []
        self.a = []
        self.b = []

        for pickles in self.all_pickles:
            self.scores.append([[] for _ in range(len(pickles))])
            self.times.append([[] for _ in range(len(pickles))])
            self.prediction_times.append([[] for _ in range(len(pickles))])
            self.prediction_files.append([[] for _ in range(len(pickles))])
            # Set up freeze thaw parameters
            self.alpha.append(3)
            self.beta.append(1)
            self.scale.append(1)
            self.log_noise.append(np.log(0.0001))
            self.x_scale.append(1)
            self.x_ell.append(0.001)
            self.a.append(1)
            self.b.append(1)

        self.bounds = [[2, 4],
                  [0.01, 5],
                  [0.1, 5],
                  [np.log(0.0000001), np.log(0.001)],
                  [0.1, 10],
                  [0.1, 10],
                  [0.33, 3],
                  [0.33, 3]]

    def think(self, action, next_state):
        """
        Process the current state to update models
        """
        self.remaining_time -= self.env.time_for_each_action
        self.time_budget = self.env.max_number_of_steps
        plot=False
        start = time.time()              # Reset the counter
        j = action.item()
        i = 0

        model_scores = [next_state[0][0].tolist()[j]]
        model_times = [next_state[1][0].tolist()[j]*self.env.max_number_of_steps]

        # Add some jitter to make the GPs happier
        # FIXME - this can be fixed with better modelling assumptions
        for k in range(len(model_scores)):
            model_scores[k] += 0.0005 * np.random.normal()

        self.scores[j][i] += model_scores

        self.times[j][i] +=  model_times
        # Save adjusted time corresponding to prediction
        self.prediction_times[j][i].append(self.times[j][i][-1])

        if self.start_thinking:
            y_mean = [None] * len(self.all_pickles)
            y_covar = [None] * len(self.all_pickles)
            predict_mean = [None] * len(self.all_pickles)
            t_star = [None] * len(self.all_pickles)
#             remaining_time = self.time_budget - (time.time() - start)
            for (j, pickles) in enumerate(self.all_pickles):
                # Run freeze thaw on data
                t_kernel = ft_K_t_t_plus_noise
                # x_kernel = cov_iid
                x_kernel = cov_matern_5_2
                # m = np.zeros((len(pickles), 1))
                m = 0.5 * np.ones((len(pickles), 1))
                t_star[j] = []
                # Subsetting data
                times_subset = copy.deepcopy(self.times[j])
                scores_subset = copy.deepcopy(self.scores[j])
                for i in range(len(pickles)):
                    if len(times_subset[i]) > 50:
                        times_subset[i] = list(np.array(times_subset[i])[[int(np.floor(k))
                                                                  for k in np.linspace(0, len(times_subset[i]) - 1, 50)[1:]]])
                        scores_subset[i] = list(np.array(scores_subset[i])[[int(np.floor(k))
                                                                    for k in np.linspace(0, len(scores_subset[i]) - 1, 50)[1:]]])

                for i in range(len(pickles)):

                    t_star[j].append(np.linspace(self.times[j][i][-1], self.times[j][i][-1] + self.remaining_time, 50))
                # Sample parameters
                xx = [self.alpha[j], self.beta[j], self.scale[j], self.log_noise[j], self.x_scale[j], self.x_ell[j]]
                # logdist = lambda xx: ft_ll(m, times_subset, scores_subset, x[j], x_kernel, dict(scale=xx[4]), t_kernel,
                #                               dict(scale=xx[2], alpha=xx[0], beta=xx[1], log_noise=xx[3]))
                logdist = lambda xx: ft_ll(m, times_subset, scores_subset, self.x[j], x_kernel, dict(scale=xx[4], ell=xx[5]), t_kernel,
                                              dict(scale=xx[2], alpha=xx[0], beta=xx[1], log_noise=xx[3]))
                xx = slice_sample_bounded_max(1, 10, logdist, xx, 0.5, True, 10, self.bounds)[0]
                self.alpha[j] = xx[0]
                self.beta[j] = xx[1]
                self.scale[j] = xx[2]
                self.log_noise[j] = xx[3]
                self.x_scale[j] = xx[4]
                self.x_ell[j] = xx[5]
    #                 print(xx)
                # Setup params
                x_kernel_params = dict(scale=self.x_scale[j], ell=self.x_ell)
                t_kernel_params = dict(scale=self.scale[j], alpha=self.alpha[j], beta=self.beta[j], log_noise=self.log_noise[j])
                y_mean[j], y_covar[j] = ft_posterior(m, times_subset, scores_subset, t_star[j], self.x[j], x_kernel, x_kernel_params, t_kernel, t_kernel_params)
                # Also compute posterior for already computed predictions
                # FIXME - what if prediction times has empty lists
                predict_mean[j], _ = ft_posterior(m, times_subset, scores_subset, self.prediction_times[j], self.x[j], x_kernel, x_kernel_params, t_kernel, t_kernel_params)
            # Identify predictions thought to be the best currently
            best_mean = -np.inf
            best_model_index = None
            best_time_index = None
            best_pickle_index = None
            for (j, pickles) in enumerate(self.all_pickles):
                for i in range(len(pickles)):
                    if max(predict_mean[j][i]) >= best_mean:
                        best_mean = max(predict_mean[j][i])
                        best_model_index = i
                        best_pickle_index = j
                        best_time_index = np.argmax(np.array(predict_mean[j][i]))
    #             print('Best pickle index : %d' % best_pickle_index)
    #             print('Best model index : %d' % best_model_index)
    #             print('Best time : %f' % prediction_times[best_pickle_index][best_model_index][best_time_index])
    #             print('Estimated performance : %f' % predict_mean[best_pickle_index][best_model_index][best_time_index])
            # Save these predictions to the output dir
    #             print('Saving predictions')
            self.predict_counter += 1
            # Pick best candidate to run next
            best_current_value = best_mean
            best_pickle_index = None
            best_model_index = -1
            best_acq_fn = -np.inf
            for (j, pickles) in enumerate(self.all_pickles):
                for i in range(len(pickles)):
                    mean = y_mean[j][i][-1]
                    std = np.sqrt(y_covar[j][i][-1, -1] - np.exp(self.log_noise[j]))
                    acq_fn = trunc_norm_mean_upper_tail(a=best_current_value, mean=mean, std=std) - best_current_value
                    if acq_fn >= best_acq_fn:
                        best_acq_fn = acq_fn
                        best_model_index = i
                        best_pickle_index = j

            if len(self.models_to_be_run) == 0:
                self.models_to_be_run.append(best_pickle_index)

            # Plot curves
            if plot:
                # TODO - Make this save to temp directory
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_title('Learning curves')
                ax.set_xlabel('Time (seconds)')
                # ax.set_xscale('log')
                ax.set_ylabel('Score')
                label_count = 0
                for j in range(len(self.all_pickles)):
                    for i in range(len(scores[j])):
                        ax.plot(times[j][i], scores[j][i],
                                color=colorbrew(label_count),
                                linestyle='dashed', marker='o',
                                label=str(label_count))
                        ax.plot(t_star[j][i], y_mean[j][i],
                                color=colorbrew(label_count),
                                linestyle='-', marker='')
                        ax.fill_between(t_star[j][i], y_mean[j][i].ravel() - np.sqrt(np.diag(y_covar[j][i]) - np.exp(log_noise[j])),
                                                   y_mean[j][i].ravel() + np.sqrt(np.diag(y_covar[j][i]) - np.exp(log_noise[j])),
                                        color=colorbrew(label_count),
                                        alpha=0.2)
                        label_count += 1
                leg = ax.legend(loc='best')
                leg.get_frame().set_alpha(0.5)
                plt.show()

    def select_action(self, state, evaluate=False):
        """
        Output the next action
        """
        action = self.models_to_be_run.pop(0)
        if len(self.models_to_be_run)==0:
            self.start_thinking = True
        return torch.tensor(action)
