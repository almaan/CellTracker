#!/usr/bin/env python3

import os
import os.path as osp
import argparse as arp

import model as m
import utils as ut
from utils import iprint,eprint
from numbers import Number

import pandas as pd
import cv2
import yaml

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from typing import *



def fit(gmphd : m.GMPHD,
        obs : pd.DataFrame,
        t0 : int = 0,
        )->None:



    times = np.unique(obs["time"].values)
    times = np.sort(times)

    for t in tqdm.tqdm(times):
        pos = obs["time"].values == t
        gmphd.update(obs[["x","y"]].values[pos,:])
        gmphd.prune()
        gmphd.update_trajectory()



def run_celltrack(obs : pd.DataFrame,
                  model_params : Dict[str,Any],
                  t0 : int = 0,
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

    ut.banner()

    prs = arp.ArgumentParser()
    sub_prs = prs.add_subparsers(dest ="command")
    aa = prs.add_argument

    run_prs = sub_prs.add_parser("run")
    ana_prs = sub_prs.add_parser("analyze")

    run_aa = run_prs.add_argument
    ana_aa = ana_prs.add_argument

    run_aa("-z","--input_data",required = True)
    run_aa("-mp","--model_parameters",required = True)
    run_aa("-t0","--initial_time",
           type = int,
           default = 0,
           )
    run_aa("-o",
           "--out_dir",
           default = "/tmp",
           )

    run_aa("-tag","--tag",
           default = "",
           )

    ana_aa("-r","--result",
           type = str,
           )

    ana_aa("-a","--animate",
       action = "store_true",
       default = False,
       )

    ana_aa("-o","--out_dir",
           default = "/tmp",
           )

    ana_aa("-img","--images",
           nargs = "+",
           )

    ana_aa("-tag",
           "--tag",
           default ="",
           )

    ana_aa("-ms","--marker_size",
           default = 5.0,
           type = float
           )

    ana_aa("-d","--delay",
           default = 10,
           type = int,
           )

    ana_aa("-kf","-keep_frames",
           default = False,
           type = bool,
           action ="store_true",
           )


    args = prs.parse_args()


    if args.command == "run":
        iprint("Reading model parameters from : {}".format(args.model_parameters))
        with open(args.model_parameters) as f:
            model_parameters = yaml.load(f, Loader=yaml.FullLoader)

        for p,v in model_parameters.items():
            if isinstance(v,str):
                model_parameters[p] = eval(v)
            if isinstance(v,dict):
                for sp,sv in v.items():
                    if isinstance(sv,str):
                        v[sp] = eval(sv)

        iprint("Reading observational data from : {}".format(args.input_data))
        obs_data = pd.read_csv(args.input_data,
                            sep = "\t",
                            header = 0,
                            index_col = 0,
                            )

        iprint("Running celltracker")
        results = run_celltrack(obs = obs_data,
                                model_params = model_parameters,
                                t0 = args.initial_time,
                                )
        iprint("Completed celltracker")

        tag = (args.tag + "-" if args.tag != "" else args.tag)

        out_fn = osp.join(args.out_dir,
                                tag + "cell-track-res.tsv",
                                )
        results.to_csv(out_fn,
                       sep = "\t",
                       )

        iprint("Saved results to : {}".format(out_fn))


    if args.command == "analyze":
        iprint("Entering analysis module")
        iprint("Using results file : {}".format(args.result))
        results = pd.read_csv(args.result,
                              sep = "\t",
                              header = 0,
                              index_col = 0,
                              )
        if args.tag == "":
            tag = osp.basename(args.results).split("-")[0] + "-"
            if tag == "cell":
                tag = ""
        else:
            tag = args.tag + "-"

        if args.animate:
            iprint("Animating results")
            from sys import platform
            if platform.lower() != "linux":
                eprint("OS not supported for animation")
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
                iprint("Initating animation")
                ut.animate_trajectories(results,
                                        images = img_pths,
                                        out_dir = args.out_dir,
                                        tag = tag,
                                        marker_size = args.marker_size,
                                        delay = args.delay,
                                        save_frames = args.keep_frames,
                                        )
                iprint("Completed animation. Results saved to : {}".format(args.out_dir))

if __name__ == "__main__":
    cli()
