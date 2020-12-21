#!/usr/bin/env python3


import cv2
import pandas as pd
import argparse as arp
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

from typing import Optional,Any,Dict,List


def get_centers(img : np.ndarray,
                kernel_size : int = 3,
                sigma : float = 10,
                thrs_prms : Optional[Dict[str,Any]] = None,
                hugh_prms : Optional[Dict[str,Any]] = None,
                )->pd.DataFrame:


    assert len(img.shape) == 2,\
        "image must be in grayscale"


    thrs_prms = (thrs_prms if thrs_prms is not None else {})
    hugh_prms = (hugh_prms if hugh_prms is not None else {})


    tmp = cv2.GaussianBlur(img,
                           ksize = (kernel_size,
                                    kernel_size),
                           sigmaX = sigma)


    _,tmp = cv2.threshold(tmp,
                          thresh = thrs_prms.pop("thresh",0),
                          maxval = thrs_prms.pop("maxval",255),
                          type = cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,
                          **thrs_prms,
                          )

    tmp = cv2.HoughCircles(tmp,
                           method = cv2.HOUGH_GRADIENT,
                           dp = hugh_prms.pop("dp",1),
                           minDist = hugh_prms.pop("minDist",20),
                           param1=hugh_prms.pop("param1",5),
                           param2=hugh_prms.pop("param2",10),
                           minRadius=hugh_prms.pop("minRadius",0),
                           maxRadius=hugh_prms.pop("maxRadius",50),
                           **hugh_prms,
                           )

    tmp = pd.DataFrame(tmp[0,:,[0,1]].T,
                       columns = ["x","y"],
                       )

    return tmp




def generate_table(img_pths : List[np.ndarray],
                   t0 : int = 0,
                   )->pd.DataFrame:


    res = list()

    for _t,pth in enumerate(img_pths):

        t = t0 + _t

        tmp = cv2.imread(pth,0)
        tmp = get_centers(tmp)
        tmp["time"] = t

        index  = ["t_{}_obs_{}".format(t,k) for k in range(tmp.shape[0])]
        tmp.index = pd.Index(index)

        res.append(tmp)


    res = pd.concat(res)

    return res



if __name__ == "__main__":

    prs = arp.ArgumentParser()
    aa = prs.add_argument


    aa("-i",
       "--images",
       nargs = "+",
       required = True,
       )


    aa("-o",
       "--out_dir",
       required = True,
       )

    aa("-tg",
       "--tag",
       default = "traj"
       )

    aa("-t0",
       "--time",
       default = 0,
       )

    aa("-p",
       "--include_image",
       action = "store_true",
       default = False,
       )

    aa("-ms",
       "--marker_size",
       default = 20,
       type = float,
       )

    args = prs.parse_args()

    res = generate_table(args.images,
                         t0 = args.time,
                         )


    res.to_csv(osp.join(args.out_dir,
                        args.tag + ".tsv"),
               sep = "\t",
               header = True,
               index = True,
               )


    if args.include_image:
        for t,pth in enumerate(args.images):
            tmp = cv2.imread(pth)
            fig,ax = plt.subplots(1,1)
            ax.imshow(tmp)
            ax.axis("off")

            pos = res["time"].values == t

            ax.scatter(res.x[pos].values,
                       res.y[pos].values,
                       s = args.marker_size,
                       c = "red",
                       )

            base = ".".join(osp.basename(pth).split(".")[0:-1])
            fig.savefig(osp.join(args.out_dir,"marked_" + base + ".png"))
            plt.close("all")



