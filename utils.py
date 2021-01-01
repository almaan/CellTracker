
import numpy as np
from scipy.stats import mode as get_mode

import cv2
import PIL
from PIL import Image,ImageDraw


from typing import List,Dict,Optional,Tuple,Union,Iterable
from numbers import Number

import pandas as pd
from operator import attrgetter,itemgetter

import subprocess as sp
import os.path as osp

import matplotlib.pyplot as plt


import tempfile

def iprint( s : str)->None:
    print("[INFO] : {}".format(s))

def eprint( s : str)->None:
    print("[ERROR] : {}".format(s))

# from model import Comp
def weighted_mode(ws : np.ndarray,
                  vals : np.ndarray,
                  )->Union[int,float]:

    v_dict : Dict[int,float] = dict()

    for v,w in zip(vals,ws):
        if v in v_dict.keys():
            v_dict[v] += w
        else:
            v_dict[v] = w

    return max(v_dict.items(),
               key=itemgetter(1))[0]


def covariance_matrix_check(S : np.ndarray,
                            precision : int = 3,
                            dim : int = 2,
                            )->bool:

    is_ok = (S.round(precision) == S.round(precision).T).all() and\
        ((S >= 0 ).flatten().all()) and\
        (S.shape[0] == dim)

    return is_ok



class Counter:
    def __init__(self,
                val : int = 0,
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


def mvneval(mu : np.ndarray,
            S : np.ndarray,
            x : np.ndarray,
            )->float:

    delta = x - mu
    y = np.exp(-0.5 * np.dot(np.dot(delta.T,inv2d(S)),delta))
    y /= np.sqrt(det2d(S)) * 2.0 * np.pi

    return float(y)

# def sampleComp( componensts : List[Comp],
#                 size : int = 1,
#                 )->np.ndarray:

#     # inspired by https://stackoverflow.com/a/4266562/13371547

#     sample = lambda x : np.random.multivariate_normal(loc = x.mu,
#                                                       scale = x.S,
#                                                       size = size,
#                                                       )
#     u = np.ranndom.random()
#     p = 0

#     for i,comp in enumerate(components):
#         p += comp.w

#         if p >= u:
#             return sample(comp)
#         elif i == len(components):
#             return sample(comp)
#         else:
#             raise Error


def format_estimates( ests : List[np.ndarray],
                      times : Iterable,
                      )->pd.DataFrame:


    res = list()

    for t,est in zip(ests,times):

        _tmp = pd.DataFrame(est,
                            columns = pd.Index(["x","y"]),
                            )
        _tmp["time"] = t

        res.append(_tmp)

    res = pd.concat(res)

    return res



def format_trajs(trajs : List[np.ndarray],
                 times : List[np.ndarray],
                 # t_end : int,
                 )->pd.DataFrame:



    res = list()

    for k,(time,traj) in enumerate(zip(times,trajs)):


        _tmp = pd.DataFrame(traj,
                            columns = ["x","y"],
                            )

        _tmp["time"] = time
        _tmp["cell"] = k

        res.append(_tmp)

    res = pd.concat(res)

    return res


def animate_trajectories(traj_res : pd.DataFrame,
                         images : List[str],
                         out_dir : str,
                         tag : str,
                         delay : int = 10,
                         color_by_id : bool = True,
                         save_frames: bool = True,
                         **kwargs,
                         )->None:

    times = np.unique(traj_res["time"].values)
    times = np.sort(times)
    n_zeros = int(np.floor(np.log10(times[-1])))

    n_times = times.shape[0]
    main_marker_size = kwargs.get("marker_size",2)

    uni_ids = np.unique(traj_res.cell)
    color_dict = {i:tuple(np.random.randint(low=0,
                                            high=255,
                                            size = 3,
                                            dtype = int))\
                  for i in uni_ids}


    with tempfile.TemporaryDirectory() as tmpdir:
        for current_t in range(n_times):
            img = cv2.imread(images[current_t])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            draw = ImageDraw.Draw(img)

            start = max(0,times[current_t]-delay)
            end = times[current_t]
            for k,t in enumerate(range(start,end+1)):
                is_time  = (traj_res["time"].values == t).flatten()
                crds = traj_res[["x","y"]].values[is_time,:]
                ids = traj_res.cell.values[is_time]

                delta = (end -start)
                delta = (1 if delta == 1 else delta)
                size_fraction = (k / delta)
                for p,i in zip(crds,ids):
                    r = main_marker_size * size_fraction / 2
                    fill = (color_dict[i] if color_by_id else (255,0,0))
                    draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=fill)

            frame = "0"+"0"*int(n_zeros - np.floor(np.log10(t + 0.1))) +\
                str(times[current_t])

            out_fn = osp.join(tmpdir,"frame_{}.png")\
                        .format(frame)

            img.save(out_fn,"PNG")

        cmd = ["convert",
               "-delay", "20",
               "-loop","0",
               osp.join(tmpdir,"*.png"),
               osp.join(out_dir,tag + "cell-track-animation.gif"),
               ]


        sp.call(cmd)
        if save_frames:
            from shutil import copytree,rmtree
            frame_dir  = osp.join(out_dir,"frames")
            if osp.exists(frame_dir):
                rmtree(frame_dir)
            copytree(tmpdir,frame_dir)



def banner()->None:
    string = "\n".join(["____ ____ _    _    ___ ____ ____ ____ _  _ ____ ____ ",
                        "|    |___ |    |     |  |__/ |__| |    |_/  |___ |__/ ",
                        "|___ |___ |___ |___  |  |  \ |  | |___ | \_ |___ |  \ ",
                        ])
    print()
    print(string)
    print()
