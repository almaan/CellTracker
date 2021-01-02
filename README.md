# Multiobject Tracking of Biological Cells using a GM-PHD filter 

This repository contains code for an implementation of the GM-PHD filter
proposed by _Vo and Ma_ ([REF](https://ieeexplore.ieee.org/document/1710358)),
with some slight modifications, designed to enable multiobject tracking of cells
from brightfield images. While code to extract cell positions - the
measurements - is provided, this is utilize a very simplistic strategy which
will not provide optimal results.

This implementation also constitutes the project part of the course FEL3320
(_Applied Estimation_), given in during the autumn of 2020 at KTH (Royal Institute of
Technology).


# Usage
The main program is `celltracker.py` which performs the state estimation based
on the observations. In order to apply `celltracker` to a data set of, it first
needs to be formatted into the expected output; outlined below.

## Input format 

The expected input to `celltracker` is a tab separated file with three required
columns (_x,y_ and _time_). The _x_ and _y_ fields gives the spatial coordinates
of an observation, and the time field at which point in time this observation
was made. See the example below:

|           |      x |     y |   time |
|:----------|-------:|------:|-------:|
| t_0_obs_0 |  164.5 | 567.5 |      0 |
| t_0_obs_1 |  280.5 | 300.5 |      0 |
| t_0_obs_2 |  408.5 | 170.5 |      0 |
| t_0_obs_3 | 1041.5 | 407.5 |      0 |
| t_0_obs_4 |  382.5 | 639.5 |      0 |


Given a set of brightfield images this table can easily be generated using the
`img2obs.py` script provided, using the following commands:

```sh
$> python3 ./img2obs.py -i IMAGE_PATHS -o /tmp/cellid2 --tag "example"

```

To also save the processed images and the identified cell centers, instead run:

```sh
$> python3 ./img2obs.py -i IMAGE_PATHS -o /tmp/cellid2 --tag "example" --include_processed_image  --mark_images

```

## Model Specifications

Once the data has been processed and the data cast into the proper input format,
running `celltracker`, only one thing remains to be done: specification of the
model. To avoid a cluttered command with plenty of different parameters, this is
done via an auxiliary configuration file (a YAML file): More precisely the
should be structured as follows:

```yaml

F: np.eye(2) # transition matrix
pD: 0.4 # detection probability
pS: 0.8 # survival probability
S: 10 * np.eye(2) # covariance for initial components
Q: 20 * np.eye(2) # process noise covariance
R: 20 * np.eye(2) # measurement noise covariance
clutter: 1e-6 # clutter (Kappa) parameter
thrs_T: 1e-3 # truncation threshold
thrs_U: 2 # merging threshold
J_max: 100 # maximum allowable number of gaussian components
spawn_params: # spawn parmaters (set to None to exclude spawning)
  N: 5 # number of new components from each existing
  w: 1e-6 # weights assigned to new components
  d: np.zeros(2) # bias in linear transformation
  Q: 10 * np.eye(2) # covariance matrix for new components
  F: np.eye(2) # transition matrix
birth_params: # birth parameters (set to None to exclude birth)
  N : 10 # number of new components
  w: 1e-8 # weights for new components
  S: 10 * np.eye(2) # covariance matrix for new components

```

_NOTE_ : `numpy` syntax is supported, and can be used to specify covariance matrices etc.

Chose a set of appropriate parameters and construct your own model
specification, or use one of the pre-existing found in the `configs` folder.

## Running celltracker

With the formatted input data and model design file, you are now ready to `run`
`celltracker`. To do this, simply do:

```sh
$> python3 ./celltrack.py run -z INPUT_FILE -mp CONFIG_FILE -t0 0 -o OUT_DIR  --tag "example"
```

Once you've run the model, you can also visualize the results using the `analyze` module, this will produce an animation (only supported on Linux systems). To generate said animation, do:

```sh

$> python3 ./celltrack.py analyze -r RESULTS_FILEK -o OUT_DIR  --tag "example" --animate --images IMAGE_DIR

```

Where `RESULTS_FILE` is the file generated from the previous run, and
`IMAGE_DIR` holds the images, `IMAGE_DIR` can also be `IMAGE_PTHS`, i.e. several
paths to the images. Note, it's expected that the images are named such that
sorting by time and name are equivalent actions. To also save the frames -- from
which the animation is compiled -- add the `--keep_frames` flag to the command
above, and these will be deposited into a subfolder of `OUT_DIR` named `frames`.


# Examples

As an example we can use the Dataset _"HeLa cells stably expressing H2b-GFP"_  from [celltrackingchallenge.net](http://celltrackingchallenge.net/2d-datasets/). The first subset of raw images consist of 93 brightfield images:

![raw images](imgs/original.png)

When processed with `img2obs.py`, we end up with the following result:

![processed images](imgs/segmented.jpg)

Running this first through `celltracker run` and then `celltracker analyze` we get:

![animation](imgs/example-001.gif)
