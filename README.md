# benthic-ensemble-paper

[![DOI](https://zenodo.org/badge/362996014.svg)](https://zenodo.org/badge/latestdoi/362996014)

This repository provides the code for the paper "Using ensemble methods to improve the robustness of deep learning for image classification in marine environments".

Documentation for using this repo as follows:

```
# install the requirements
pip install -r requirements.txt

# downloads the image patches to the data folder
sh get_data.sh

# constructs the dataset i.e. does the train/val splits and mean image creation
python make_dataset.py

# trains the ensembles, 15 of them
sh train_ensembles.sh

# inferences the test data with each model
sh run_ensemble_inference.sh

# performs the blur shift augmentation inferencing
sh run_ensemble_inference_blur.sh

# performs the colour shift augmentation inferening
sh run_ensemble_inference_colour.sh

# subsambles the ensembles for each of the methods using in the paper, Single, MC Dropout and Ensmeble, currently size 10
python subsample_models.py

```

Once all the above has been performed, the results are explored in the notebook - benthic_ensemble_paper.ipynb 
