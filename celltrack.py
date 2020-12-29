#!/usr/bin/env python3

import os
import os.path as osp
import argparse as arp

import model as m
import utils as ut
from numbers import Number

import pandas as pd
import cv2
import yaml

import numpy as np
import matplotlib.pyplot as plt

from typing import *



def fit(gmphd : m.GMPHD,
        obs : pd.DataFrame,
        t0 : int = 0,
        )->None:



    times = np.unique(obs["time"].values)
    times = np.sort(times)

    for t in times[(t0+1)::]:
        pos = obs["time"].values == t
        gmphd.update(obs[["x","y"]].values[pos,:])
        gmphd.prune()
        gmphd.update_trajectory()



def run_celltrack(obs : pd.DataFrame,
                  model_params : Dict[str,Any],
                  t0 : int = 0,
                  # birth_prms : Optional[Dict[str,Any]] = None,
                  # spawn_prms : Optional[Dict[str,Any]] = None,
                  )->pd.DataFrame:


    required_columns = ["x","y","time"]
    has_columns = all([x in obs.columns for x\
                       in required_columns
                       ])

    assert has_columns,\
        "columns {} must be present in"\
        " observational data"\
        .format(",".join(required_columns))



    init_crd = obs["time"].values == int(t0)
    init_crd = obs[["x","y"]].values[init_crd,:]


    if isinstance(model_params["S"],Number):
        S = model_params.pop("S") * np.eye(2)
    elif isinstance(model_params["S"],np.ndarray):
        S = model_params.pop("S")
    else:
        S = 10 * np.eye(2)


    model_params["initial"] = [m.Comp(w = 1.0 / init_crd.shape[0],
                                     mu = init_crd[k,:],
                                     S = S,
                                     comp_id = k,
                                     t = 0,
                                     ) for k in range(init_crd.shape[0])]


    # model_prms["birth_prms"] = birth_prms
    # model_prms["spawn_prms"] = spawn_prms

    gmphd = m.GMPHD(**model_params)
    fit(gmphd,
        obs = obs,
        t0 = t0,
        )


    trajs,times = gmphd.compile_trajectory()

    res = ut.format_trajs(trajs,
                          times,
                          )

    return res


def cli()->None:

    prs = arp.ArgumentParser()
    aa = prs.add_argument


    aa("-z","--input_data",required = True)
    aa("-mp","--model_parameters",required = True)
    aa("-t0","--initial_time",
       type = int,
       default = 0,
       )
    aa("-a","--animate",
       action = "store_true",
       default = False,
       )

    aa("-od","--out_dir",
       default = "/tmp",
       )

    aa("-img","--images",
       nargs = "+",
       )

    aa("-tag","--tag",
       default = "",
       )


    args = prs.parse_args()



    with open(args.model_parameters) as f:
        model_parameters = yaml.load(f, Loader=yaml.FullLoader)

    for p,v in model_parameters.items():
        if isinstance(v,str):
            model_parameters[p] = eval(v)
        if isinstance(v,dict):
            for sp,sv in v.items():
                if isinstance(sv,str):
                    v[sp] = eval(sv)


    obs_data = pd.read_csv(args.input_data,
                           sep = "\t",
                           header = 0,
                           index_col = 0,
                           )

    results = run_celltrack(obs = obs_data,
                            model_params = model_parameters,
                            t0 = args.initial_time,
                            )

    tag = (args.tag + "-" if args.tag != "" else args.tag)

    results.to_csv(osp.join(args.out_dir,
                            tag + "cell-track-res.tsv",
                            ))


    if args.animate:
        from sys import platform
        if platform.lower() != "linux":
            print("OS not supported for animation")
        else:
            if osp.isdir(args.images[0]):
                image_ext = ["png",
                            "tiff",
                            "tif",
                            "jpg",
                            "jpeg",
                            "gif",
                            "bmp"]
                is_image = lambda x : x.split(".")[-1] in image_ext
                img_pths = list(filter(is_image,os.listdir(args.images[0])))
                img_pths = [osp.join(args.images[0],p) for p in img_pths]
            else:
                img_pths = args.images

            img_pths.sort()

            ut.animate_trajectories(results,
                                    images = img_pths,
                                    out_dir = args.out_dir,
                                    tag = tag,
                                    )

if __name__ == "__main__":
    cli()
