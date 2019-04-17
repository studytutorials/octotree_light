# Scripts
The scripts in this directory are to help with the automation of running tests.

These scripts assume the datasets are contained in a single directory e.g.

* `~/Datasets/icl-nuim_living_room_1/...`
* `~/Datasets/icl-nuim_living_room_2/...`

and so on.



## Initialization
On the first use there is some setup to be performed. 

### Downloading datasets
You can bulk download some ICL-NUIM and TUM RGB-D datasets using the
`download_datasets.sh` Bash script. 

### `systemsettings.py`
Setup the correct paths

* `BIN_PATH` : Path to the binaries (the default value should be fine).
* `RESULTS_PATH` : Path to store the results at.
* `DATASETS_PATH` : Path to the folder containing the datasets (in the example
  above this would be `~/Datasets`).

### `datasets.py`
Most of the settings in here are fixed for a particular dataset. You may need
to change the paths to the datasets. The paths are relative to `DATASETS_PATH`.



## Evaluating on datasets
The script `run_benchmarks.py` is used to evaluate supereight on multiple
datasets. You can edit the lists over which the for loops iterate to change the
datasets to evaluate, the volume resolutions used and the map type (currently
only 'sdf' or 'ofusion').

