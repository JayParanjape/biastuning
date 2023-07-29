# AdaptiveSAM
This repository contains the code for **AdaptiveSAM: Towards Efficient Tuning of SAM for Surgical Scene Segmentation**

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create --file=biastuning_env.yaml
conda activate biastuning_env
```

## General file descriptions
- data_transforms/*.py - data transforms defined here for different datasets.
- data_utils.py - functions to generate dataloaders for different datasets
- model.py - model architectures defined here
- train.py - code for general training, common to all datasets
- train_baselines.py - driver code for generating results on baselines described in the paper.
- driver_scratchpad.py - driver code for training models. 
- eval/*/generate_predictions.py - code for generating results for a given folder
- model_biastuning.yml - config file for defining various model hyperparameters for AdaptiveSAM
- model_baselines.yml - config file for different baseline models
- config_<dataset_name>.yml - config file for defining various dataset related hyperparameters
  
## Example Usage for Training
```
python driver_scratchpad.py --model_config model_biastuning.yml --data_config config_cholec8k.yml --save_path "./temp.pth"
```
## Example Usage for Evaluation
```
cd eval/endovis

python generate_predictions.py --model_config config_model_test.yml --data_config config_endovis_test.yml --data_folder <path to image folder> --gt_path <path to ground truth images folder> --save_path "./temp_results" --pretrained_path <path to model>
```

## Citation
```
To be filled
```