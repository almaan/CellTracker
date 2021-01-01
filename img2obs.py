#!/usr/bin/env python3


import cv2
import pandas as pd
import argparse as arp
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

from typing import *


def get_centers(img : np.ndarray,
                kernel_size : int = 21,
                sigma : float = 10,
                thrs_prms : Optional[Dict[str,Any]] = None,
                hugh_prms : Optional[Dict[str,Any]] = None,
                include_processed_image : bool = False,
                )->Tuple[pd.DataFrame,Optional[np.ndarray]]:


    assert len(img.shape) == 2,\
        "image must be in grayscale"

    thrs_prms = (thrs_prms if thrs_prms is not None else {})
    hugh_prms = (hugh_prms if hugh_prms is not None else {})


    tmp = cv2.GaussianBlur(img,
                           ksize = (kernel_size,
                                    kernel_size),
                           sigmaX = sigma)


    _,tmp = cv2.threshold(tmp,
                          thresh = thrs_prms.pop("thresh",200),
                          maxval = thrs_prms.pop("maxval",255),
                          type = cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,
                          **thrs_prms,
                          )

    res = cv2.HoughCircles(tmp,
                           method = cv2.HOUGH_GRADIENT,
                           dp = hugh_prms.pop("dp",1),
                           minDist = hugh_prms.pop("minDist",20),
                           param1=hugh_prms.pop("param1",5),
                           param2=hugh_prms.pop("param2",10),
                           minRadius=hugh_prms.pop("minRadius",0),
                           maxRadius=hugh_prms.pop("maxRadius",50),
                           **hugh_prms,
                           )

    res = pd.DataFrame(res[0,:,[0,1]].T,
                       columns = ["x","y"],
                       )

    if include_processed_image:
        return res,tmp
    else:
        res,None




def generate_table(img_pths : List[np.ndarray],
                   t0 : int = 0,
                   include_processed_image : bool = False,
                   )->Tuple[pd.DataFrame,Optional[List[np.ndarray]]]:


    res : List[pd.DataFrame] = list()
    processed_images : List[np.ndarray] = list()

    for _t,pth in enumerate(img_pths):

        t = t0 + _t

        tmp = cv2.imread(pth,0)
        tmp,proc_img = get_centers(tmp,
                                   include_processed_image = include_processed_image)
        tmp["time"] = t

        index  = ["t_{}_obs_{}".format(t,k) for k in range(tmp.shape[0])]
        tmp.index = pd.Index(index)

        res.append(tmp)
        if include_processed_image:
            processed_images.append(proc_img)


    res = pd.concat(res)

    if include_processed_image:
        return res,processed_images
    else:
        return res,None



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

    aa("-mi",
       "--mark_images",
       action = "store_true",
       default = False,
       )

    aa("-pp",
       "--include_processed_images",
       action = "store_true",
       default = False,
       )

    aa("-ms",
       "--marker_size",
       default = 20,
       type = float,
       )

    args = prs.parse_args()

    res,processed_images = generate_table(args.images,
                                          args.time,
                                          args.include_processed_images,
                                          )


    res.to_csv(osp.join(args.out_dir,
                        args.tag + ".tsv"),
               sep = "\t",
               header = True,
               index = True,
               )

    if args.include_processed_images:
        for t,tmp in enumerate(processed_images):

            base = ".".join(osp.basename(args.images[t]).split(".")[0:-1])

            if args.mark_images:
                tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2RGB)
                pos = (res.time == t)
                for p in res[["x","y"]].values[pos,:]:
                    tmp = cv2.circle(tmp,
                                     tuple(p),
                                     radius = 5,
                                     color = (255,0,0),
                                     thickness = -1)

            cv2.imwrite(osp.join(args.out_dir,"marked_" + base + ".png"),
                        tmp)



