from argparse import ArgumentParser
import configparser
import re

from azureml.core.workspace import Workspace
from azureml.core import Experiment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.dnn import PyTorch
from azureml.core.runconfig import MpiConfiguration
from azureml.core.dataset import Dataset

from constants import TRAIN_TAG, TEST_TAG, PARENT_MODEL_TAG, SCRIPT_TAG, MSG_TAG

def _get_name_and_version(source):
    """ Get source (dataset or model) name and version.
        @param: source - format: "NAME:VERSION" or "NAME=VERSION" or "NAME"
        @return (name, version)
    """
    name, version = '', ''
    if source:
        s = re.split(r'[:=]', source)
        name =  s[0]
        version = s[1] if len(s)>1 else 'latest'
        if version != 'latest':
            version = int(version)
    return name, version

def _get_script_params(train_cfg, cfg):
    config = configparser.ConfigParser()
    config.read(train_cfg)
    script_params = {k: v for k, v in config['DEFAULT'].items()}
    if config[cfg]:
        for k, v in config[cfg].items():
            script_params[k] = v
    return [(f"--{k}", v) for k, v in script_params.items()]

def _get_compute_target(workspace, cluster_name):
    """ Get existed compute target from AzureML.
        @param: workspace - Workspace.
        @param: cluster_name - Cluster name.
        @return: compute_target
    """
    compute_target = None
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
        print(f"Found existing compute target: {cluster_name}")
    except ComputeTargetException:
        raise Exception(f"Error! compute target: {cluster_name} is not found. Please create it on AzureML or use existed one.")
    return compute_target

def _get_source_dataset(workspace, source, as_mount=True):
    name, version = _get_name_and_version(source)
    dataset = Dataset.get_by_name(
        workspace = workspace,
        name      = name,
        version   = version
    ).as_named_input(name)
    return dataset.as_mount() if as_mount else dataset.as_download()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='alpr-ocr-deprecated', help='Experiment name to display in Experiments tab.')
    parser.add_argument('--cluster_name',    type=str, default='gpu-cluster', help='Name of cluster. Please refer to AzureML dashboard')
    parser.add_argument('--train_cfg',       type=str, default='config/default_azureml.ini', help='ALPR OCR Model training parameters config file path.')
    parser.add_argument('--cfg',             type=str, required=True,  help="Model type: GlamdringV3, GlamdringV10, ...")
    parser.add_argument('--trainroot',       type=str, required=True,  help='Training data name and version. Format `name:version` or `name` (default version=latest)')
    parser.add_argument('--valroot',         type=str, required=False, help='Testing data name and version. Format `name:version` or `name` (default version=latest)')
    parser.add_argument('--as_mount',        type=int, default=1, help='Input datasets as_mount or as_download')
    parser.add_argument('--model',           type=str, default='', help='Registered (pre-trained) model name and version. Format `name:version` or `name` (default version=latest)')
    parser.add_argument('--node_count',      type=int, default=4, help='Number of AzureML nodes for distributed training with horovod.')
    parser.add_argument('--process_count',   type=int, default=4, help='Process count per node for distributed training with horovod.')
    parser.add_argument('--msg',             type=str, default='', help='Experiment message.')
    args = parser.parse_args()

    script_params = _get_script_params(args.train_cfg, args.cfg)
    script_params['--model_path'] = args.model
    script_params['--cfg'] = args.cfg

    workspace = Workspace.from_config()
    compute_target = _get_compute_target(
        workspace    = workspace, 
        cluster_name = args.cluster_name
    )
    
    script_params['--trainroot'] = _get_source_dataset(workspace, source=args.trainroot, as_mount=args.as_mount)
    script_params['--valroot'] = ''
    if args.valroot == args.trainroot:
        script_params['--valroot'] = script_params['--trainroot']
    elif args.valroot:
        script_params['--valroot'] = _get_source_dataset(workspace, source=args.valroot, as_mount=args.as_mount)

    estimator = PyTorch(
        source_directory       ='../', 
        entry_script           = 'main.py', 
        script_params          = {k: v for k, v in script_params.items() if v != ''},
        node_count             = args.node_count, 
        process_count_per_node = args.process_count,
        compute_target         = compute_target, 
        distributed_training   = MpiConfiguration(), 
        framework_version      ='1.5', 
        use_gpu                = True,
        shm_size               = "16g",
        # conda_packages='environment.yml'
        environment_variables={
            'NCCL_DEBUG': 'DEBUG',
            'AZUREML_DATASET_HTTP_RETRY_COUNT': 10000,
            'NCCL_RINGS': ','.join(map(str, range(args.node_count)))
        },
        conda_packages=[
            'opencv',
            'imutils',
            'imgaug',
        ],
        pip_packages=[
            'coloredlogs',
            'PyYAML',
            'click',
            'tensorboardX',
            'lmdb',
            'tqdm',
            'joblib',
            'terminaltables',
            'tensorly',
            'https://dataprepdownloads.azureedge.net/install/wheels/test-M3ME5B1GMEM3SW0W/23936626/ship/azureml_dataprep-2.4.0.dev0+3eff1fb-py3-none-any.whl',
            'https://dataprepdownloads.azureedge.net/install/wheels/test-M3ME5B1GMEM3SW0W/23936626/ship/azureml_dataprep_rslex-1.2.0.dev0+3eff1fb-cp36-cp36m-manylinux2010_x86_64.whl',
            'https://dataprepdownloads.azureedge.net/install/wheels/test-M3ME5B1GMEM3SW0W/23936626/ship/azureml_dataprep_native-0.1.15+dev-cp36-cp36m-linux_x86_64.whl'
        ],
        # # pip_requirements_file='requirements_azureml.txt'
    )
    experiment = Experiment(workspace, name=args.experiment_name)
    run = experiment.submit(
        estimator, 
        tags = {
            TRAIN_TAG: args.trainroot,
            TEST_TAG: args.valroot,
            PARENT_MODEL_TAG: args.model,
            SCRIPT_TAG: str(__file__),
            MSG_TAG: args.msg
        }
    )
    run.wait_for_completion(show_output=True)
