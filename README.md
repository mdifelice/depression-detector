It allows to train datasets inside the `data` folder. The datasets structure and which algorithms are supported are indicated in the `data/metadata.json` file.

It uses the `pandas`, `sklearn`, `scipy` and `numpy` packages.

The script can be called via bash and their arguments are:

* `-a`: Whether to print metrics from all algorithms or only the best one.
* `-c`: Whether to print pre-processing information.
* `-d <dataset_id>`: Which dataset to train. Several can be indicated, separated by commas. The dataset ID is defined by the position the occupy in the metadata definition. By default, all datasets are trained.
* `-e <level>`: Whether to print debug information. The level can be 1 to 4.
* `-f`: Whether to force tuning. By default, tuning is only performed if the default training does not meet the metrics threshold.
* `-h`: Whether to generate charts. They are saved in the `charts` folder.
* `-i <value>`: How many cross validation folds to perform when tuning. Default value is 5.
* `-l <scaler>`: Which scaler to use. Possible values are `standard`, `minmax` or `robust`. Default scaler is `standard`.
* `-m <model_id>`: Which model to train. Several can be indicated, separated by commas. The model ID is defined by the key used in the metadata file.
* `-n`: Whether to perform tune.
* `-o <value>`: Indicates the number of rows a dataset must have to perform oversampling. By default is 0, meaning no oversampling will be used.
* `-r`: Turbo mode. It uses all available processors.
* `-s <test_ratio>': Test ratio. By default is 0.3.
* `-t`: Whether to train.
* `-u`: Whether to perform unsupervised analysis.
* `-v <value>`: How many cross validations folds to perform. Default value is 5.
* `-w <random_seed>`: The random seed to initiate algorithms. Default value is 123.
* `-x <model_id>`: Which model to exclude from train. Several can be indicated, separated by commas. The model ID is defined by the key used in the metadata file.
* `-y <value>`: How many tune iterations to perform. Default value is 10.
