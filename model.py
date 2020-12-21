#!/usr/bin/env python3

import numpy as np
from typing import List,Dict,Optional,Tuple,Union
from copy import deepcopy
from operator import attrgetter,itemgetter

from numbers import Number

from scipy.stats import mode as get_mode


class Counter:
    def __init__(self,
                val = 0,
                )->None:

        self.val = val

    def __call__(self,)->int:
        self.val += 1
        return self.val -1


def det2d( X : np.ndarray,
           )-> float:

    return X[0,0]*X[1,1] - X[0,1]*X[1,0]

def inv2d( X : np.ndarray,
           )->np.ndarray:

    invX = np.zeros(X.shape)
    invX[0,0] = X[1,1]
    invX[1,1] = X[0,0]
    invX[0,1] = -X[0,1]
    invX[1,0] = -X[1,0]
    invX /= det2d(X)
    return invX


class Comp:

    # write for optimal performance in 2D

    def __init__(self,
                 w : float,
                 mu : np.ndarray,
                 S : np.ndarray,
                 comp_id : int,
                 t : int = 0,
                 lineage : Optional[List[Tuple[int,int]]] = None,
                 )->None:



        assert (S.round(3) == S.round(3).T).all() and\
            ((S >= 0 ).flatten().all()),\
            "covariance must be symmetric and positive"

        self.__w = w
        self.__mu = mu
        self.__S = S

        self.normalizer = np.sqrt(det2d(self.__S)) * 2.0 * np.pi
        self.invS = inv2d(self.__S)

        self.id = comp_id

        self.lineage = []
        if lineage is not None:
            if isinstance(lineage,tuple):
                self.lineage += [lineage]
            else:
                self.lineage += lineage

        # self.lineage.append((t,self.id))


    def __len__(self,)->int:
        return self.w.shape[0]

    def eval(self,x : np.ndarray,
             )->float:

        delta = x - self.__mu
        y = np.exp(-0.5 * np.dot(np.dot(delta.T,self.invS),delta))
        y /= self.normalizer

        return y

    @property
    def mu(self,)->np.ndarray:
        return self.__mu
    @mu.setter
    def mu(self, mu,)->None:
        assert mu.shape[0] == 2,\
            "wrong length of my"
        self.__mu = mu

    @property
    def S(self,)->np.ndarray:
        return self.__S

    @S.setter
    def S(self,S)->None:
        self.__S = S
        self.normalizer = np.sqrt(det2d(self.__S)) * 2.0 * np.pi
        self.invS = inv2d(self.__S)

    @property
    def w(self,)->float:
        return self.__w
    @w.setter
    def w(self,w)->None:
        assert w >= 0,\
            "weights must be n.n"

        self.__w = float(w)





def mvneval(mu : np.ndarray,
            S : np.ndarray,
            x : np.ndarray,
            )->float:

    delta = x - mu
    y = np.exp(-0.5 * np.dot(np.dot(delta.T,inv2d(S)),delta))
    y /= np.sqrt(det2d(S)) * 2.0 * np.pi

    return float(y)


def sampleComp( componensts : List[Comp],
                size : int = 1,
                )->np.ndarray:
    # inspired by https://stackoverflow.com/a/4266562/13371547

    sample = lambda x : np.random.multivariate_normal(loc = x.mu,
                                                      scale = x.S,
                                                      size = size,
                                                      )
    u = np.ranndom.random()
    p = 0

    for i,comp in enumerate(components):
        p += comp.w

        if p >= u:
            return sample(comp)
        elif i == len(components):
            return sample(comp)
        else:
            raise Error



class GMPHD:

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
                 birth_params : Optional[Dict[str,Number]] = None,
                 spawn_params : Optional[Dict[str,Number]] = None,
                 t : int = 0,
                 ) -> None:


        self.t = t

        self.mix = initial
        self.pD = pD
        self.pS = pS
        self.F = F
        self.Q = Q
        self.R = R
        self.clutter = clutter


        if birth_params is None:
            self.birth_params = None
        else:
            birth_param_names = ["N","w","mu","S"]
            check_birth_params = all([x in birth_params.keys() for x in birth_param_names])
            self.birth_params = (birth_params if check_birth_params else None)


        if spawn_params is None:
            self.spawn_params = None
        else:
            spawn_param_names = ["N","w","F","d","Q"]
            check_spawn_params = all([x in spawn_params.keys() for x in spawn_param_names])
            self.spawn_params = (spawn_params if check_spawn_params else None)

        if thrs_T is None:
            self.thrs_T = 1.0 / len(self.mix)
        else:
            self.thrs_T = thrs_T

        self.thrs_U = thrs_U
        self.J_max = J_max

        self.I = np.eye(2)

        self.genid = Counter(val = max([x.id for x in self.mix]))

        self._track_state = []
        self._track_lineage = []
        self._track_idxs = []




    def breed(self,
              )->List[Comp]:


        born = []

        if self.birth_params is not None:
            for k in range(self.birth_params.get("N")):
                _comp = Comp(w = self.birth_params("w"),
                             mu = self.birth_params("mu"),
                             S = self.birth_params("S"),
                             id = self.genid(),
                             t = self.t,
                            )

                born.append(_comp)


        return born

    def spawn(self,
              )->List[Comp]:


        spawned = []
        if self.spawn_params is not None:
            for k in range(self.spawn_params["N"]):
                for comp in self.mix:
                    w_k_km1 = comp.w * self.spawn_params["w"]
                    mu_k_km1 = self.spawn_params["d"] + np.dot(self.spawn_params["F"],
                                                            comp.mu,
                                                            )
                    S_k_km1 = self.spawn_params["Q"] + np.dot(np.dot(self.spawn_params["F"],
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
                                        ))

        return spawned

    def update(self,
               Z : np.ndarray,
               )->None:

        self.t += 1

        # Step 1
        born = self.breed()

        predicted = self.mix + born

        spawned = self.spawn()

        predicted += spawned


        # Step 2

        for comp in self.mix:
            comp.w = self.pS * comp.w
            comp.mu = np.dot(self.F,comp.mu)
            comp.S = self.Q + np.dot(np.dot(self.F,comp.S),self.F.T)


        # Step 3

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

        # Step 4
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


            scale = float(1.0 / (sum([c.w for c in newmixpart]) + self.clutter))

            for comp in newmixpart:
                comp.w *= scale


            newmix.extend(newmixpart)

        self.mix = newmix


    def dist_U(self,
               j : int,
               i : int,
               )->float:

        delta = self.mix[i].mu - self.mix[j].mu
        inv_S = self.mix[i].invS

        d = np.dot(np.dot(delta.T,inv_S),delta)

        return d

    def prune(self,
              )->None:

        self.mix = [comp for comp in self.mix if comp.w > self.thrs_T]

        w_sum_old = sum([comp.w for comp in self.mix])

        set_I = set(range(len(self.mix)))
        max_iter = len(set_I)
        newmix = []

        l = 0

        while (len(set_I) > 0) or (l > max_iter):

            l +=1

            j = [(i,self.mix[i].w) for i in set_I]
            j.sort(key = lambda x: x[1])
            j = j[0][0]

            set_L = [i for i in set_I if self.dist_U(j,i) <= self.thrs_U ]

            set_L = set(set_L)

            if len(set_L) > 0:

                w_l = sum([self.mix[i].w for i in set_L])

                mu_l = 1.0 / w_l * sum([self.mix[i].mu * self.mix[i].w for i in set_L])

                S_l = []

                L_vals = np.array([self.mix[i].lineage[-1][1] for i in set_L])
                L_ws = np.array([self.mix[i].w for i in set_L])
                lin = weighted_mode(L_ws,L_vals)


                for i in set_L:
                    _delta = mu_l - self.mix[i].mu
                    _term = self.mix[i].w * (self.mix[i].S + np.dot(_delta,_delta.T))

                    S_l.append(_term)

                S_l = 1.0 / w_l * sum(S_l)

                set_I = set_I.difference(set_L)

                newmix.append(Comp(w_l,
                                   mu_l,
                                   S_l,
                                   comp_id = self.genid(),
                                   t = self.t,
                                   lineage = (self.t -1,lin)
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

        self._strong_components = []
        for k,comp in enumerate(self.mix):
            if comp.w > 0.5:
                # for _ in range(int(round(comp.w))):
                    self._strong_components.append(k)


    def update_trajectory(self,
                          )->None:


        _states = []
        _lineages = []
        _idxs = []
        for j in self._strong_components:
            _states.append(self.mix[j].mu)
            _lineages.append(self.mix[j].lineage[-1][1])
            _idxs.append(self.mix[j].id)

        self._track_state.append(np.asarray(_states))
        self._track_lineage.append(_lineages)
        self._track_idxs.append(_idxs)


    def compile_trajectory(self,
                           )->List[np.ndarray]:


        trajs = []
        T = len(self._track_state)
        for center in range(len(self._track_state[-1])):
            _crds = list()
            k = center
            for _t in range(1,T+1):
                t = T - _t
                _crds.append(self._track_state[t][k,:])
                parent = self._track_idxs[t-1] == self._track_lineage[t][k]
                if parent.sum() > 0:
                    k = np.argmax(self._track_idxs[t-1] == self._track_lineage[t][k])
                else:
                    break

            trajs.append(np.asarray(_crds))

        return trajs


    def extractstate(self,
                     )->np.ndarray:

        # state = []
        # lineage = []
        # idxs = []

        # for j in self.strong_components:
            # state.append(self.mix[j].mu)
            # lineage.append(self.mix[j].lineage[-2][1])
            # idxs.append(self.mix[j].id)

        # return (np.asarray(state), lineage, idxs)

        return self._track_state[-1]



def weighted_mode(ws : np.ndarray,
                  vals : np.ndarray,
                  )->Union[int,float]:

    v_dict = dict()

    for v,w in zip(vals,ws):
        if v in v_dict.keys():
            v_dict[v] += w
        else:
            v_dict[v] = w

    return max(v_dict.items(), key=itemgetter(1))[0]



