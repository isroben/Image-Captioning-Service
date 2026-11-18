import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    'configs/config.yaml',
    'src/data/preprocessing.py',
    'src/models/captioner.py',
    'src/serving/app.py',
    'tests/',
    'Dockerfile',
    'pyproject.toml'
]

for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as file:
            pass
        logging.info(f"Creaging empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists.")