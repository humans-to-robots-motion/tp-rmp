import logging
import itertools
import numpy as np
from sys import float_info

from scipy.stats import multivariate_normal as mvn


class EM(object):
    '''
    "Generalizing Robot Imitation Learning with Invariant Hidden Semi-Markov Models." Tanwani, 2018. [1]
    "Encoding the time and space constraints of a task in explicit-duration hidden Markov model." Calinon, 2011. [2]
    Expectation Maximization optimizer for TP-HSMM
    NOTE: This implementation of EM is stateful, i.e. dependent on self variables. Hence all API calls should only be used with public functions!
    '''
    logger = logging.getLogger(__name__)

    def __init__(self, demos, **kwargs):
        # record variables of demo
        self._demos = demos
        self._num_demos = len(demos)
        self._frame_names = self.demos[0].frame_names
        self._manifold = self.demos[0].manifold
        self._dim_M = self.demos[0].dim_M
        self._dt = self.demos[0].dt
        self._num_points = sum([demo.length for demo in self.demos])
        # params
        self._num_comp = kwargs.get('num_comp', 20)
        self._regularization = kwargs.get('regularization', 1e-5)
        self._min_iter = kwargs.get('min_iter', 5)
        self._max_iter = kwargs.get('max_iter', 200)
        self._accuracy_ll = kwargs.get('accuracy_ll', 1e-5)
        self._num_comp_per_tag = kwargs.get('num_comp_per_tag', None)  # this will override num_comp if not None
        self._topology = kwargs.get('topology', None)
        self._with_tag = kwargs.get('with_tag', False)
        self.reset()

    def reset(self):
        '''Reset function'''
        self.gamma = []
        self.zeta = []
        self.duration_prob = None
        self.end_states = np.zeros((self._num_comp,))
        self.max_duration = None
        self._initialize()

    def optimize(self):
        EM.logger.info(f'Start training TP-HSMM with {self._num_comp} components.')
        av_ll = 0
        # EM-Iteration
        for i in range(self._max_iter):
            av_ll_prev = av_ll
            # E and M step:
            av_ll, self.gamma, self.zeta = self._E_step()
            self._M_step()
            # stopping criterion:
            if (i > self._min_iter) & (av_ll - av_ll_prev < self._accuracy_ll):
                EM.logger.info(f'Training converged after {i} steps')
                break
            elif i == self._max_iter - 1:
                EM.logger.warn(f'Maximum number of iterations {self._max_iter} reached!')
        # make duration model
        self._duration_model()
        # remove self transitions to ensure smooth execution # NOTE: this is somewhat handcrafted
        for k in range(self._num_comp):
            self.trans_prob[k, k] = 0
            if self.end_states[k] > 0:  # TODO: do we need to consider the case where end state "may" lead to other state?
                self.trans_prob[k, np.where(self.trans_prob[k, :] < 1e-5)] = 0  # ensure end state leads to no where else
            else:
                self.trans_prob[k, :] /= self.trans_prob[k, :].sum()

    def _initialize(self):
        self.tag_to_comp_map = {}
        if self._with_tag:
            demo_tags = {}
            i = 0
            if self._num_comp_per_tag is not None:
                comp_split = [0]
                for tag, nb_comp in self._num_comp_per_tag:
                    demo_tags[tag] = i
                    i += 1
                    comp_split = np.append(comp_split, comp_split[-1] + nb_comp)
                self._num_comp = comp_split[-1]
            else:
                for demo in self.demos:
                    if demo.tag not in demo_tags:
                        demo_tags[demo.tag] = i
                        i += 1
                comp_split = np.linspace(0, self._num_comp, len(demo_tags) + 1, dtype=int)
            for tag, tag_idx in demo_tags.items():
                self.tag_to_comp_map[tag] = np.arange(comp_split[tag_idx], comp_split[tag_idx + 1])
        EM.logger.info(f'Tagging: {self.tag_to_comp_map}')
        self.mvns = []
        self.pi = np.zeros((self._num_comp,))
        cluster = [{frame: np.zeros((self._dim_M, 0)) for frame in self._frame_names} for _ in range(self._num_comp)]  # init cluster point holder
        for demo in self.demos:
            if self._with_tag:
                num_comp = comp_split[demo_tags[demo.tag] + 1] - comp_split[demo_tags[demo.tag]]
            else:
                num_comp = self._num_comp
            equal_split = np.linspace(0, demo.length, num_comp + 1, dtype=int)
            for k in range(num_comp):
                if self._with_tag:
                    comp_id = comp_split[demo_tags[demo.tag]] + k
                else:
                    comp_id = k
                for frame in self._frame_names:
                    cluster[comp_id][frame] = np.column_stack((cluster[comp_id][frame], demo.traj_in_frames[frame]['traj'][:, equal_split[k]:equal_split[k + 1]]))
        for k in range(self._num_comp):
            num_cluster_points = cluster[k][self._frame_names[0]].shape[1]
            self.pi[k] = float(num_cluster_points) / self._num_points
            self.mvns.append({})
            for frame in self._frame_names:
                self.mvns[k][frame] = self._manifold.normal_distribution(cluster[k][frame], regularization=self._regularization)
        # init trans prob
        if self._topology is not None:
            self._topology = np.array(self._topology)
            self.trans_prob = self._topology / self._topology.sum(axis=1, keepdims=True)  # normalizing
        else:
            self.trans_prob = np.ones((self._num_comp, self._num_comp)) / self._num_comp

    def _E_step(self):
        ll = 0
        zeta_all = np.zeros((0, self._num_comp, self._num_comp))
        gamma_all = []
        # compute probability variables for each demo and accumulate them
        for demo in self.demos:
            obsrv_prob = self._observation_model(demo)
            ll_temp, alpha, scale = self._forward(demo, obsrv_prob)
            beta = self._backward(obsrv_prob, scale)
            gamma, zeta = self._prob_model(alpha, beta, obsrv_prob)
            zeta_all = np.append(zeta_all, zeta, 0)
            gamma_all.append(gamma)
            ll += ll_temp
        av_ll = ll / self._num_comp
        return av_ll, gamma_all, zeta_all

    def _M_step(self):
        # update Gaussians
        self.mvns, gamma_sum = self._update_mvns()
        # update transition probability
        self.trans_prob = np.sum(self.zeta, axis=0) * self._topology / gamma_sum
        # update priors
        self.pi = np.zeros((self._num_comp,))
        for m in range(self._num_demos):
            self.pi[:] += self.gamma[m][0, :]
        self.pi /= self._num_demos

    def _observation_model(self, demo):
        '''observation probabilities as product over probabilities in each frame.'''
        obsrv_prob = np.ones((demo.length, self._num_comp))
        for k in range(self._num_comp):
            for frame in self._frame_names:
                if self.mvns[k][frame] is not None:
                    obsrv_prob[:, k] *= self.mvns[k][frame].pdf(demo.traj_in_frames[frame]['traj'])
        return obsrv_prob

    def _duration_model(self):
        ds = [[]] * self._num_comp  # list to collect the durations (occurring in gamma) in each component
        self.end_states = np.zeros((self._num_comp,))  # contains prob of which each component is an end state
        for m in range(self._num_demos):
            s = np.argmax(self.gamma[m], axis=1)
            # get durations
            d = 1
            for i in range(1, s.shape[0]):
                if s[i] == s[i - 1]:  # still in this state
                    d = d + 1
                else:  # state transition
                    ds[s[i - 1]].append(d)
                    d = 1
            ds[s[-1]].append(d)  # take in all the end of demonstration as duration
            self.end_states[s[-1]] += 1 / self._num_demos  # adjust end state probability of this demonstrations end state
        # Set maximal duration according to rule of thumb in footnote 3 in [2]
        self.max_duration = int(max([g.shape[0] for g in self.gamma]) * 3 / self._num_comp)  # TODO: check this
        self.duration_prob = np.zeros((self._num_comp, self.max_duration))
        for k in range(self._num_comp):
            mvn_duration = mvn(np.mean(ds[k]), np.var(ds[k]) + 1)
            for d in range(self.max_duration):
                self.duration_prob[k, d] = mvn_duration.pdf(d)
            self.duration_prob[k, :] /= (self.duration_prob[k, :].sum() + float_info.min)

    def _forward(self, demo, obsrv_prob):
        alpha = np.zeros((demo.length, self._num_comp))
        scale = np.ones((demo.length, 1))
        alpha[0, :] = self.pi * obsrv_prob[0, :]
        ll = np.log(alpha[0, :].sum())
        scale[0, :] /= (alpha[0, :].sum() + float_info.min)
        alpha[0, :] *= scale[0, :]
        for t in range(1, demo.length):
            alpha[t, :] = self.trans_prob.T.dot(alpha[t - 1, :]) * obsrv_prob[t, :]
            ll += np.log(alpha[t, :].sum() + float_info.min)
            scale[t, :] /= (alpha[t, :].sum() + float_info.min)
            alpha[t, :] *= scale[t, :]
        return ll, alpha, scale

    def _backward(self, obsrv_prob, scale):
        num_points = obsrv_prob.shape[0]
        beta = np.zeros((num_points, self._num_comp))
        beta[-1, :] = 1 * scale[-1, :]
        # compute beta
        for t in range(num_points - 2, -1, -1):
            beta[t, :] = self.trans_prob.dot(obsrv_prob[t + 1, :] * beta[t + 1, :])
            beta[t, :] = np.minimum(beta[t, :] * scale[t, :], float_info.max)
        return beta

    def _prob_model(self, alpha, beta, obsrv_prob):
        num_points = alpha.shape[0]
        zeta = np.zeros((num_points - 1, self._num_comp, self._num_comp))
        # compute probability variables
        gamma = alpha * beta
        iter_comps = range(self._num_comp)
        for t in range(num_points - 1):
            for m, n in itertools.product(iter_comps, iter_comps):
                zeta[t, m, n] = (alpha[t, m] * self.trans_prob[m, n] * obsrv_prob[t + 1, n] * beta[t + 1, n] * self._topology[m, n])
            zeta[t, :, :] /= (zeta[t, :, :].sum() + float_info.min)
            gamma[t, :] /= (gamma[t, :].sum() + float_info.min)
        gamma[-1, :] /= (gamma[-1, :].sum() + float_info.min)
        return gamma, zeta

    def _update_mvns(self, init_mean=None):
        if init_mean is None:
            init_mean = [{frame: self.mvns[k][frame].mean for frame in self.mvns[k]} for k in range(self._num_comp)]
        mvns = []
        gamma_sum = np.zeros((self._num_comp, 1))
        h = []
        for k in range(self._num_comp):
            h.append(np.zeros(0))
            mvns.append({})
            points_in_frames = {frame: np.zeros((self._dim_M, 0)) for frame in self._frame_names}
            # compute the weighted mean over all data points of all demonstrations in each frame.
            for m, demo in enumerate(self.demos):
                for frame in self._frame_names:
                    points_in_frames[frame] = np.column_stack((points_in_frames[frame], demo.traj_in_frames[frame]['traj']))
                h[k] = np.append(h[k], self.gamma[m][:, k])
                gamma_sum[k, 0] += np.sum(self.gamma[m][0:-1, k])
            # compute and add to mvns[k] the Gaussian for each frame
            for frame in self._frame_names:
                mvns[k][frame] = self._manifold.normal_distribution(points_in_frames[frame], regularization=self._regularization, init_mu=init_mean[k][frame], weights=h[k])
        return mvns, gamma_sum

    @property
    def demos(self):
        return self._demos

    @property
    def num_comp(self):
        return self._num_comp

    @property
    def model_parameters(self):
        model_params = {
            'mvns': self.mvns,
            'pi': self.pi,
            'trans_prob': self.trans_prob,
            'duration_prob': self.duration_prob,
            'max_duration': self.max_duration,
            'end_states': self.end_states,
            'gamma': self.gamma,
            'zeta': self.zeta,
            'tag_to_comp_map': self.tag_to_comp_map,
            'dim_M': self._dim_M,
            'dt': self._dt
        }
        return model_params
