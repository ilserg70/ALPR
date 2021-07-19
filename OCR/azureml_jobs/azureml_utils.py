import re
import logging
from pathlib import Path
import torch

# import azureml.core
from azureml.core.run import Run
from azureml.core.model import Model
from azureml.core import Dataset

# Model tags keys
from azureml_jobs.constants import TRAIN_TAG, TEST_TAG, ACC_TAG, TAGS, EXPERIMENT_TAG

logger = logging.getLogger(__name__)

""" This is singleton like module.
"""

EXP_RUN    = Run.get_context() # Current experiment Run
print(f"Run_ID: {EXP_RUN._run_id}") # OfflineRun_d5c670ad-929b-40f0-a1ea-baac0315bcd9
ONLINE   = 'OfflineRun' not in EXP_RUN._run_id

WORKSPACE  = None
EXP_NAME   = None
RUN_NUMBER = None
EXP_TAGS   = None

if ONLINE:
    WORKSPACE  = EXP_RUN._experiment._workspace
    EXP_NAME   = EXP_RUN._experiment._name
    RUN_NUMBER = EXP_RUN._run_number
    EXP_TAGS   = EXP_RUN.get_tags() # {TRAIN_TAG: .., TEST_TAG: .., PARENT_MODEL_TAG: ..., SCRIPT_TAG: ..}

def input_datasets(name):
    return EXP_RUN.input_datasets[name] if ONLINE else name

def log_metric(name, value, dscr=''):
    """ Logging metric.
            @param: name - Metric name
            @param: value - Metric value
            @param: dscr - Description
    """
    EXP_RUN.log(name, value, dscr)

def get_name_and_version(source):
    """ Get source (dataset or model) name and version.
        @param: source - format: "NAME:VERSION"
        @return (name, version)
    """
    name, version = source, None
    if source:
        s = re.split(r'[:=]', source)
        name =  s[0]
        if len(s)>1:
            version = s[1]
        if version != 'latest':
            version = int(version)
    return name, version

def find_max_accuracy_model(model_name):
    """ Finding registered model with best accuracy and the same name and 
        built on the same train/test datasets
            @param: model_name - Model name
            @return: (max_acc, model_dscr)
    """
    max_acc, model = 0, None
    try:
        # Get all versions of registered models with name: model_name on given workspace.
        mm = Model.list(workspace=WORKSPACE, name=model_name)
        for m in mm:
            train_id = m.tags.get(TRAIN_TAG, None) # NAME:VERSION
            test_id = m.tags.get(TEST_TAG, None)   # NAME:VERSION
            acc = float(m.tags.get(ACC_TAG, '0.0%').replace('%',''))
            if train_id==EXP_TAGS[TRAIN_TAG] and test_id==EXP_TAGS[TEST_TAG] and acc>max_acc:
                max_acc, model = acc, m
        if model:
            logger.info(f"Found best registered model {model.id} with accuracy: {max_acc:.2f}%")
    except Exception as e:
        logger.error(f"Error! Finding max accuracy model: {model_name} - {e}")
    return max_acc, model

def get_model(model_id):
    """ Downloading Model from registered models to current experiment dir.
            @param: model_id - Model ID, format: NAME:VERSION
            @return: model_path - Model weights path in current experiment dir
                IMPORTANT! Don't do Path(model_path).resolve(), it must be relative path in experiment dir otherwise
                   file not found exception will be raised at the moment of model weights uploading by torch.
    """
    model_path = None
    try:
        name, version = get_name_and_version(model_id)
        model_path = Model.get_model_path(name, version)
        logger.info(f"Model: {model_id} was downloaded from registered models to: {model_path}")
    except Exception as e:
        logger.error(f"Error! Downloading model: {model_id} from registered models - {e}")
    return model_path

def register_model(weights_path, model_name, descr=None, tags={}):
    """ Register a new model on AzureML. Make sure that no model of previous version 
        with better accuracy on the same datasets.
            @param: weights_path - Model weights path.
            @param: model_name - Model name.
            @param: descr - Model description.
            @param: tags - dict of model tags, example: {ACC_TAG: '87.56%', ...}
            @return: True if registered
    """
    def _mk_model_tags(tags):
        _tags = tags.copy()
        try:
            for k, v in EXP_RUN.get_tags().items():
                if v != '' and k in TAGS:
                    _tags[k] = v
            _tags[EXPERIMENT_TAG] = f"{EXP_NAME} Run {RUN_NUMBER}"
        except Exception as e:
                logger.error(f"Error! Getting tags from experiment run - {e}")
        return _tags
    
    try:
        logger.info(f"Start registration of a new model: {model_name} with accuracy: {tags[ACC_TAG]} ...")
        
        # Make sure that no model of previous version with better accuracy.
        max_acc, _model = find_max_accuracy_model(model_name)
        _acc = float(tags[ACC_TAG].replace('%',''))
        if _acc <= max_acc:
            logger.info(f"Warning! Skip registration. Found model {_model.id} with better accuracy: {_model.tags[ACC_TAG]}.")
            return False

        # IMPORTANT! Don't do Path(weights_path).resolve(), it must be relative path
        # on AzureML experiment, otherwise file not found exception will be raised.
        p = Path(weights_path)

        if p.exists() and p.is_file():
            model = EXP_RUN.register_model(
                model_name  = model_name,
                model_path  = str(weights_path),
                tags        = _mk_model_tags(tags),
                description = (descr if descr else model_name),
                model_framework = Model.Framework.PYTORCH,
                model_framework_version = torch.__version__
            )
            logger.info(f'Model: {model.name}, version: {model.version} is registered.')
    except Exception as e:
            logger.error(f"Failed to register model: {model_name} - {e}")
            return False
    return True
