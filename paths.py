import os
import subprocess
import yaml


IMAGE_DIR = subprocess.check_output(
    "find ~ -type d -name depth_images",
    shell=True,
    executable="/bin/bash"
).decode().strip()

WORKING_DIR = subprocess.check_output(
    "find ~ -type d -name snake_fyp",
    shell=True,
    executable="/bin/bash"
).decode().strip()

DATASET_CSV = f'{WORKING_DIR}/dataset.csv'
SPLIT_DIR = f'{WORKING_DIR}/split'

CHECKPOINT_DIR = f'{WORKING_DIR}/checkpoints'
MISC_DIR = f'{WORKING_DIR}/misc'

dirs = [CHECKPOINT_DIR, MISC_DIR]

# with open(f'{WORKING_DIR}/config.yaml') as file:
#     params = yaml.safe_load(file)

def create_dirs():

    for dir in dirs:
        if os.path.exists(dir):
            pass
        else:
            os.mkdir(dir)
            print(f'Created {dir} folder...')
            if dir == MISC_DIR:
                configs = {
                            'status': 'not_started',
                            'best_val_loss': 10.0,
                            'total_elapsed_seconds': 0,
                            'epochs': 0
                        }
                with open(f'{MISC_DIR}/misc.yaml', 'w') as file:
                    yaml.safe_dump(configs, file)

create_dirs()


