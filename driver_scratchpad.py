import argparse
import yaml
from data_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config_tmp.yml',
                        help='config file path')

    args = parser.parse_args()

    return args

def main_datautils():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    dataset_dict, dataset_sizes, label_dict = get_data(config, tr_folder_start=0, tr_folder_end=78, val_folder_start=78, val_folder_end=104)
    print(len(dataset_dict['train']))
    print(dataset_dict['train'][0])

if __name__ == '__main__':
    #for checking data_utils
    main_datautils()