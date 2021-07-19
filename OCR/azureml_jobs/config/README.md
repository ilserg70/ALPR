# Configuration files

This directory contains configuration files for ALPR OCR preprocess/training/validation

## Preprocess pipeline configuration

[preprocess.yml](preprocess.yml) - conda dependencies for [submit_preprocess_job.py](../submit_preprocess_job.py) and [preprocess](../../preprocess)

## Environment configuration

[training.yml](training.yml) - conda dependencies for: [submit_training_job.py](../submit_training_job.py) and [main.py](../../main.py)

Originally this file is a copy of AzureML curated environment: `ALPR-Pytorch1.7-Cuda11-OpenMpi4.1.0-py36`

To get it from AzureML use:
```python
workspace = Workspace.from_config()
curated_environments = {
    env_name: envs[env_name].python.conda_dependencies.serialize_to_string() 
    for env_name in Environment.list(workspace=workspace) if env_name.startswith("AzureML")
}
```

## ALPR OCR Model configuration

[training.ini](training.ini) - contains ALPR OCR Model training parameters which are used in the script: [main.py](../../main.py)

