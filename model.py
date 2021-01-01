#!/usr/bin/env python3

import numpy as np
from scipy.stats import mode as get_mode


from typing import *
from numbers import Number

from operator import attrgetter,itemgetter

from utils import *


class Comp:
    
    """Component in Gaussian Mixture

    Components used to construct a Gaussian Mixture,
    has a mean, covariance matrix and weight associated
    with it. Each component is assigned an id and has a
    lineage attribute in order to track the progress.

    These components only support 2D data, while less flexible
    this allows for better performance as many calculations
    can be simplified in the 2D case, e.g., calculation of
    the inverse and determinant.


    Parameters:
    -----------

    w : float
        weight in mixture
    mu : np.ndarray
        mean of 2D Gaussian 
    S : np.ndarray
        covariance matrix of 2D Gaussian
    comp_id: int
        component's id
    lineage : Optional[List[Tuple[int,int]]]
        A list of tuples or single tuple in the format
        (timepoint,id of parent)

    """

    def __init__(self,
                 w : float,
                 mu : np.ndarray,
                 S : np.ndarray,
                 comp_id : int,
                 t : int = 0,
                 lineage : Optional[Tuple[int,int]] = None,
                 spawned : bool = False,
                 )->None:


        # check that S is proper cov.matrix
        assert covariance_matrix_check(S),\
            "covariance must be a 2x2 positive symmetric matrix"

        # check that weights are non-negative
        assert w >= 0,\
            "weights must be non-negative"
        # check dimensionality of mean
        assert mu.shape[0] == 2,\
            "mean must be 2D"

        # set component attributes
        self.__w = w
        self.__mu = mu
        self.__S = S

        # is component spawned
        self.spawned = spawned

        # pre-compute inverse
        self.invS = inv2d(self.__S)
        # set id of component. Used for
        # lineage tracing
        self.id = comp_id

        # prepare for lineage storage
        self.lineage = []
        if lineage is not None:
            if isinstance(lineage,tuple):
                self.lineage += [lineage]
            else:
                self.lineage += lineage

    @property
    def mu(self,)->np.ndarray:
        return self.__mu

    @mu.setter
    def mu(self, mu,)->None:
        assert mu.shape[0] == 2,\
            "mean must be 2D"
        self.__mu = mu

    @property
    def S(self,)->np.ndarray:
        return self.__S

    @S.setter
    def S(self,S)->None:
        assert covariance_matrix_check(S),\
            "covariance must be a 2x2 positive symmetric matrix"
        self.__S = S
        self.invS = inv2d(self.__S)

    @property
    def w(self,)->float:
        return self.__w

    @w.setter
    def w(self,w)->None:
        assert w >= 0,\
            "weights must be non-negative"
        self.__w = float(w)

class GMPHD:

    """GM-PHD Model

    An implementation of the model presented in:
    -------------
    Title: A Closed-Form Solution for the Probability Hypothesis Density Filter
    Authors : Ba-Ngu Vo and Wing-Kin Ma
    DOI: 10.1109/TSP.2006.881190
    -------------

    Additional features have been added to allow
    lineage tracking to assess how particles
    move through time.


    initial : List[Comp]
        list with set of components to initialize model with
    pS : float
        probability of survival
    pD : float,
        probability of death
    F : np.ndarray
        state transition matrix
    Q : np.ndarray
        process noise covariance matrix
    R : np.ndarray,
        measurement noise covariance matrix
    clutter : float
        clutter parameter (kappa in original publication)
    thrs_T : float = None
        truncation threshold for components in pruning
    thrs_U : float (0.1)
        merging threshold for components in pruning
    J_max : int (50)
        maximum allowable number of Gaussian terms
    birth_params : Optional[Dict[str,Union[int,float,np.ndarray]]] (None)
        A dictionary dictating the birth process,
        containing:

          N - the number of components to add
          w - weight to assign new components
          S - covariance matrix for new components

        If set to None, no births will occur.

    spawn_params : Optional[Dict[str,Union[int,float,np.ndarray]]] (None)
        A dictionary dictating the spawn process,
        containing:

          N - the number of components to spawn from each
              existing component
          w - weight to assign spawned components
          Q - covariance matrix for new components
          d - bias term in affine transformation of component mean
          F - transition matrix to propagate mean


        If set to None, no spawning will occur.

    t : int (0)
        Time point at which data series starts

    """

    def __init__(self,
                 initial : List[Comp],
                 pS : float,
                 pD : float,
                 F : np.ndarray,
                 Q : np.ndarray,
                 R : np.ndarray,
                 clutter : float,
                 thrs_T : float = None,
                 thrs_U : float = 0.1,
                 J_max : int = 50,
                 birth_params : Optional[Dict[str,Any]] = None,
                 spawn_params : Optional[Dict[str,Any]] = None,
                 t : int = 0,
                 ) -> None:

        # set time
        self.t = t

        # check validity of parameters
        for name,prob in zip(["survival probability",
                              "death probability"],
                             [pD,pS]
                       ):
            assert prob >= 0 and prob <=1,\
                "{} must be in interval [0,1]".format(name)

        for name,M in zip(["Q","R"],
                          [Q,R]
                          ):

            assert covariance_matrix_check(M),\
                "{} is not a proper covariance matrix".format(M)

        # set attributes
        self.mix = initial
        self.pD = pD
        self.pS = pS
        self.F = F
        self.Q = Q
        self.R = R
        self.clutter = clutter


        # control that necessary entries in birth_params are
        # present
        if birth_params is None:
            self.birth_params = None
        else:
            birth_param_names = ["N","w","S"]
            check_birth_params = all([x in birth_params.keys() for x in birth_param_names])
            self.birth_params = (birth_params if check_birth_params else None)

        # control that necessary entries in spawn_params are
        # present
        if spawn_params is None:
            self.spawn_params = None
        else:
            spawn_param_names = ["N","w","F","d","Q"]
            check_spawn_params = all([x in spawn_params.keys() for x in spawn_param_names])
            self.spawn_params = (spawn_params if check_spawn_params else None)

        # set truncation threshold
        if thrs_T is None:
            self.thrs_T = 1.0 / len(self.mix)
        else:
            self.thrs_T = thrs_T

        # set merging threshold
        self.thrs_U = thrs_U
        # set maximum number of allowable components
        self.J_max = J_max
        # set identity matrix, for later use
        self.I = np.eye(2)

        # instantiate ID-generator object
        max_val : int = max([x.id for x in self.mix])
        self.genid : Counter = Counter(val = max_val)

        # set lists to hold lineage information
        self._track_state : List[np.ndarray] = []
        self._track_lineage : List[np.ndarray] = []
        self._track_idxs : List[np.ndarray] = []
        self._track_times : List[np.ndarray] = []


    def breed(self,
              Z : np.ndarray,
              )->List[Comp]:

        """Birth process

        Will sample mean coordinates randomly from inhabited
        space, birth params will be used for remaining
        attributes.

        """


        born = []

        if isinstance(self.birth_params,dict):
            n_obs = Z.shape[0]
            probs = [ sum([mvneval(c.mu,c.S,Z[ii,:]) for c in self.mix]) for ii in range(n_obs)]
            ordr = np.argsort(probs)

            for k in range(int(self.birth_params.get("N",1))):
                # _mu = sample_mu()

                _comp = Comp(w = self.birth_params["w"],
                             mu = Z[ordr[k % n_obs],:],
                             S = self.birth_params["S"],
                             comp_id = self.genid(),
                             t = self.t,
                            )

                born.append(_comp)


        return born

    def spawn(self,
              )->List[Comp]:

        """Spawning process"""


        spawned = []
        if self.spawn_params is not None:
            for k in range(self.spawn_params["N"]):
                for comp in self.mix:
                    if not comp.spawned:
                        w_k_km1 = comp.w * self.spawn_params["w"]
                        mu_k_km1 = self.spawn_params["d"] +\
                            np.dot(self.spawn_params["F"],
                                comp.mu,
                            )

                        S_k_km1 = self.spawn_params["Q"] +\
                            np.dot(np.dot(self.spawn_params["F"],
                                        comp.S,
                                        ),
                                self.spawn_params["F"].T,
                                )


                        spawned.append(Comp(w = w_k_km1,
                                            mu = mu_k_km1,
                                            S = S_k_km1,
                                            comp_id = self.genid(),
                                            t = self.t,
                                            lineage = (self.t-1,comp.id),
                                            spawned = True,
                                            ))

        return spawned

    def __len__(self,
                )->int:

        return len(self.mix)

    def update(self,
               Z : np.ndarray,
               )->None:

        """Update State

        Parameters:
        ----------
        Z : np.ndarray
            set of observations, on the form [n_obs x 2]

        """


        # Step 1 | Prediction of brith targets and spawning
        born = self.breed(Z)

        predicted = self.mix + born

        spawned = self.spawn()

        predicted += spawned


        # Step 2 | Predicition for existing targets

        for comp in self.mix:
            comp.w = self.pS * comp.w
            comp.mu = np.dot(self.F,comp.mu)
            comp.S = self.Q + np.dot(np.dot(self.F,comp.S),self.F.T)


        # Step 3 | Construction of PHD update components

        M_k_list = []
        K_k_list = []
        S_k_k_list = []

        for comp in predicted:
            M_k = self.R + comp.S

            K_k = np.dot(comp.S, inv2d(M_k))
            S_k_k = np.dot(self.I - K_k,comp.S)

            M_k_list.append(M_k)
            K_k_list.append(K_k)
            S_k_k_list.append(S_k_k)

        # Step 4 | Update
        newmix= []
        for comp in predicted:
            newmix.append(Comp(w = (1.0 - self.pD)*comp.w,
                               mu = comp.mu,
                               S = comp.S,
                               comp_id = self.genid(),
                               t = self.t,
                               lineage = (self.t-1,comp.id),
                               )

                          )

        for z in Z:
            newmixpart = []
            for j,comp in enumerate(predicted):
                w_k = self.pD * comp.w * mvneval(mu = comp.mu,
                                                 S = M_k_list[j],
                                                 x = z,
                                                 )

                mu_k = comp.mu + np.dot(K_k_list[j],(z - comp.mu))

                newmixpart.append(Comp(w = w_k,
                                       mu = mu_k,
                                       S = S_k_k_list[j],
                                       comp_id = self.genid(),
                                       t = self.t,
                                       lineage = (self.t-1,comp.id),
                                       ))


            scale = float(1.0 / (sum([c.w for c\
                                      in newmixpart]) +\
                                 self.clutter))

            for comp in newmixpart:
                comp.w *= scale


            newmix.extend(newmixpart)

        self.mix = newmix
        self.t += 1


    def dist_U(self,
               j : int,
               i : int,
               )->float:
        """Compute squared Mahalanobis distance

        helper function for pruning, used to find which
        components that should be joined togeter (if result
        is less than thrs_U).

        Parameters:
        ----------

        j : int
            index of weight with highest value
        i : int
            weight index to test for merging

        Returns:
        -------
        The value of (mu_i - mu_j)^T (P^i)^{-1}(m_i-m_j)

        """

        delta = self.mix[i].mu - self.mix[j].mu
        inv_S = self.mix[i].invS

        d = np.dot(np.dot(delta.T,inv_S),delta)

        return d

    def prune(self,
              )->None:

        """Prune components

        Based on the algorithm given in Table II
        of the original publication.

        """

        self.mix = [comp for comp in self.mix if\
                    comp.w > self.thrs_T]

        w_sum_old = sum([comp.w for comp in self.mix])

        set_I = set(range(len(self.mix)))
        max_iter = len(set_I)
        newmix = []
        l = 0
        while (len(set_I) > 0) or (l > max_iter):
            l += 1

            j = [(i,self.mix[i].w) for i in set_I]
            j.sort(key = lambda x: x[1])
            j = j[0][0]

            set_L = [i for i in set_I if \
                     (self.dist_U(j,i) <= self.thrs_U) ]

            if len(set_L) > 0:

                w_l = sum([self.mix[i].w for i in set_L])

                mu_l = 1.0 / w_l * sum([self.mix[i].mu *\
                                        self.mix[i].w for i in set_L])

                S_l = []

                L_vals = np.array([self.mix[i].lineage[-1][1] for i in set_L])
                L_ws = np.array([self.mix[i].w for i in set_L])
                lin = weighted_mode(L_ws,L_vals)


                for i in set_L:
                    _delta = mu_l - self.mix[i].mu
                    _term = self.mix[i].w * (self.mix[i].S +\
                                             np.dot(_delta,_delta.T))

                    S_l.append(_term)

                S_l = 1.0 / w_l * sum(S_l)

                set_I = set_I.difference(set_L)

                newmix.append(Comp(w_l,
                                   mu_l,
                                   S_l,
                                   comp_id = self.genid(),
                                   t = self.t,
                                   lineage = (self.t -1,int(lin))
                                   ))

        if len(newmix) >= self.J_max:
            newmix.sort(key = attrgetter("w"))
            newmix.reverse()
            newmix = newmix[0:self.J_max]

        self.mix = newmix

        w_sum_new = sum([comp.w for comp in self.mix])

        w_adj_factor = w_sum_old / w_sum_new

        for comp in self.mix:
            comp.w *= w_adj_factor

        self._update_strong_components()


    def _update_strong_components(self,
                                  )->None:
        """Update multiple-target state set

        Updates the list of components to be used
        in the multiple-target state extraction

        """

        self._strong_components = []
        for k,comp in enumerate(self.mix):
            if comp.w >= 0.5:
                # for _ in range(int(round(comp.w))):
                    self._strong_components.append(k)


    def update_trajectory(self,
                          )->None:
        """Update trajectory information

        function used to track propagation of
        objects

        """

        _states = []
        _lineages = []
        _idxs = []
        _times = []
        for j in self._strong_components:
            _states.append(self.mix[j].mu)
            _lineages.append(self.mix[j].lineage[-1][1])
            _idxs.append(self.mix[j].id)
            _times.append(self.t)

        self._track_state.append(np.asarray(_states))
        self._track_lineage.append(_lineages)
        self._track_idxs.append(_idxs)
        self._track_times.append(_times)


    def compile_trajectory(self,
                           )->Tuple[List[np.ndarray],
                                    List[np.ndarray]]:

        """Compile trajectory from model

        Returns:
        -------

        Tuple with (1) a list of trajectories, each element
        is a distinct trajectory; and (2) a list of arrays with
        coupled time points for respective trajectory in
        (1).

        """


        trajs = []
        times = []

        # T = len(self._track_state)

        # for center in range(len(self._track_state[-1])):
        #     _crds = list()
        #     _time = list()
        #     k = center
        #     for _t in range(1,T+1):
        #         t = T - _t
        #         _crds.append(self._track_state[t][k,:])
        #         _time.append(self._track_times[t][k])

        #         parent = [ x == self._track_lineage[t][k] for x\
        #                    in self._track_idxs[t-1]]

        #         if sum(parent) > 0:
        #             k = np.argmax(parent)
        #         else:
        #             break
        #     trajs.append(np.asarray(_crds))
        #     times.append(np.asarray(_time))

        # return trajs,times

        T = len(self._track_state)
        tracked = list()

        for t in range(0,T):
            _crds = list()
            _time = list()

            idxs = set(self._track_idxs[t])

            for idx in idxs:
                _crds = list()
                _time = list()

                if idx not in tracked:
                    tracked.append(idx)
                    pos = [x == idx for x in self._track_idxs[t]]
                    pos = np.argmax(pos)

                    _crds.append(self._track_state[t][pos])
                    _time.append(self._track_times[t][pos])

                    tt = t + 1
                    new_idx = idx

                    while tt < T:

                        child = [x == new_idx for x in\
                                 self._track_lineage[tt]
                                 ]

                        if sum(child) == 0:
                            break

                        child = np.argmax(child)

                        _crds.append(self._track_state[tt][child])
                        _time.append(self._track_times[tt][child])

                        new_idx = self._track_idxs[tt][child]
                        tracked.append(new_idx)

                        tt +=1

                    trajs.append(np.asarray(_crds))
                    times.append(np.asarray(_time))

        return trajs,times



    def extractstate(self,
                     )->np.ndarray:
        """Extract current state

        Based on Table III in the original publication

        Returns:
        -------
        Array with current state estimate

        """

        return self._track_state[-1]
