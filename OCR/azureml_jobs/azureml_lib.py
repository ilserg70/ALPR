from pathlib import Path
import configparser

import azureml.core
from azureml.core.workspace import Workspace
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import ComputeTarget
from azureml.telemetry import set_diagnostics_collection
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.data import OutputFileDatasetConfig
from azureml.core.datastore import Datastore
from azureml.core import Environment, Experiment, RunConfiguration, Dataset, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration

def get_script_params(cfg_file, cfg_name=None):
    """ Reading config .ini file with training/testing parameters
    @param: cfg_file - Path to config .ini file
    @param: cfg_name - [Optional] Model name for training
    @return: Dict of training/testing parameters
    """
    config = configparser.ConfigParser()
    config.read(cfg_file)
    script_params = {k: v for k, v in config['DEFAULT'].items()}
    if cfg_name and config[cfg_name]:
        for k, v in config[cfg_name].items():
            script_params[k] = v
    script_params = [(f"--{k}", v) for k, v in script_params.items()]
    return [x for p in script_params for x in p]

def check_data_version(source):
    """ Check format of 'NAME:VERSION' for registered Datasets and Models
    @param: source - 'NAME:VERSION'
    @return None or assertion error
    """
    p = source.split(':')
    assert len(p)==2 and p[1].isnumeric(), f"Define numeric version of {source}"

def print_version_set_diagnostics():
    """ Print AzureMML SDK version and set diagnostics collection
    """
    print(f"SDK version: {azureml.core.VERSION}")
    set_diagnostics_collection(send_diagnostics=False)

def get_compute_target(workspace, cluster_name):
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

def get_run_config(conda_cfg='', pip_packages=[], conda_packages=[]):
    """ Get run_config by different ways. It is used in pipeline version.
        @param: conda_cfg       - Path to conda dependencies file
        @param: pip_packages    - Pithon packages
        @param: conda_packages  - Conda packages
        @return: run_config
    """
    if conda_cfg and Path(conda_cfg).exists():
        conda_dep = CondaDependencies(conda_dependencies_file_path=conda_cfg)
    else:
        conda_dep = CondaDependencies.create(
            pip_packages   = pip_packages,
            conda_packages = conda_packages
        )
    return RunConfiguration(conda_dependencies=conda_dep)

def get_all_environments(workspace):
    """ Get AzureML environments existed on given workspace
        @param: workspace - Workspace.
        @return: dict of environment names and conda dependencies (str)
    """
    envs = Environment.list(workspace=workspace)
    return {
        env_name: envs[env_name].python.conda_dependencies.serialize_to_string() 
        for env_name in envs
    }

def get_environment(workspace, env_name, is_register_env=False, curated_env_name=None, conda_cfg = '',
    env_vars={}, pip_pkgs=[], conda_pkgs=[]):
    """ Get existed Environment by name or create a new one by different ways.
        @param: workspace - AzureML Workspace.
        @param: env_name - Environment name (of existed env. or the name for the new one).
        @param: is_register_env - flag if need to register a new environment with name env_name
        @param: curated_env_name - AzureML curated environment name started with 'AzureML'
        @param: conda_cfg - Conda dependencies file path. To create a new environment.
        @param: env_vars   | Another way to create a new environment is to define variables,
        @param: pip_pkgs   | pithon packages
        @param: conda_pkgs | and conda packages.
        @return: environment (existed or created env.)
    Documentation: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments
    Example: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch
    """
    assert not env_name.startswith('AzureML'), "Error! Environment name can not start with the prefix AzureML"
    
    envs = get_all_environments(workspace)
    if env_name in envs:
        print("------------------")
        print("Found environment:")
        print("------------------")
        print(envs[env_name])
        return Environment.get(workspace=workspace, name=env_name)

    if curated_env_name and curated_env_name.startswith('AzureML') and curated_env_name in envs:
        print("----------------------------------")
        print("Found AzureML curated environment:")
        print("----------------------------------")
        print(envs[curated_env_name])
        _env = Environment.get(workspace=workspace, name=curated_env_name).clone(env_name)
    elif conda_cfg and Path(conda_cfg).exists():
        _env = Environment.from_conda_specification(name=env_name, file_path=str(conda_cfg))
        _env.docker.enabled = True
        #_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
    else:
        _env = Environment(name=env_name)

    if env_vars:
        _env.environment_variables.update(env_vars)
    for _pkg in pip_pkgs:
        _env.python.conda_dependencies.add_pip_package(_pkg)
    for _pkg in conda_pkgs:
        _env.python.conda_dependencies.add_conda_package(_pkg)

    if is_register_env:
        _env.register(workspace=workspace)
        print(f"Registered a new environment: {env_name}")
    
    return _env

class ALPRExperiment():
    """ Abstract class of ALPR Experiment.
    """    
    def __init__(self, cluster_name, source_dir, datastore_name=None):
        print_version_set_diagnostics()
        self.workspace = Workspace.from_config()
        if datastore_name:
            self.datastore = Datastore.get(self.workspace, datastore_name)
        else:
            self.datastore = self.workspace.get_default_datastore()
        self.compute_target = get_compute_target(self.workspace, cluster_name=cluster_name)
        self.source_dir = source_dir

    def source_dataset(self, source, as_mount=True):
        """ Get input Dataset
        @param: source - Format `name:version`
        @param: as_mount - True: as mount, False: as download
        @return: Dataset
        """
        name, version = source.split(':')
        dataset = Dataset.get_by_name(
            workspace = self.workspace,
            name      = name,
            version   = int(version)
        ).as_named_input(name.replace('-','_'))
        return dataset.as_mount() if as_mount else dataset.as_download()

class ALPRPipeline(ALPRExperiment):
    """ Class of ALPR Pipeline components for data preprocessing.
    """
    def __init__(self, cluster_name, conda_cfg, source_dir, datastore_name=None, allow_reuse=True):
        super(ALPRPipeline, self).__init__(cluster_name, source_dir, datastore_name)
        
        self.run_config = get_run_config(conda_cfg=conda_cfg)
        self.allow_reuse = allow_reuse

    def pipeline_dataset(self, name):
        """ Make Pipeline Dataset
        @param: name - Dataset name
        @return Dataset
        """
        return PipelineData(
            name                 = name, 
            pipeline_output_name = name,
            datastore            = self.datastore, 
            is_directory         = True
        ).as_dataset()

    def out_dataset(self, name, as_mount=True, overwrite=False):
        """ Make OutputFileDatasetConfig
        @param: name - Dataset name
        @param: as_mount - True: as mount, False: as upload
        @param: overwrite - Overwrite flag
        @return: OutputFileDatasetConfig
        """
        ds = OutputFileDatasetConfig(
            name        = name, 
            destination = (self.datastore, name)
        )
        return ds.as_mount() if as_mount else ds.as_upload(overwrite=overwrite)

    def step_register_dataset(self, datasets, names, dscrs, tags={}):
        """ Make Pipeline step: register datasets
        @param: datasets - List of Dataset
        @param: names - Registration names
        @param: dscrs - Registration descriptions
        @return: PythonScriptStep
        """
        return PythonScriptStep(
            name                   = 'register_dataset',
            script_name            = 'register_dataset.py',
            source_directory       = self.source_dir,
            arguments              = [
                '--names', ','.join(names),
                '--dscrs', ','.join(dscrs),
                '--tags', ','.join([f"{k}:{v}" for k, v in tags.items()]),
                '--datastore_name', self.datastore.name
            ],
            inputs                 = [d.as_named_input(n) for d, n in zip(datasets, names)],
            outputs                = [],
            compute_target         = self.compute_target,
            runconfig              = self.run_config,
            allow_reuse            = self.allow_reuse,
        )

    def step_extract_plates(self, input, input_name, output, min_score=0.1, process_count=8):
        """ Make Pipeline step: extract plate images from frames
        @param: input - Input Dataset
        @param: input_name - Input Dataset NAME:VERSION
        @param: output - Output Dataset
        @param: min_score - Min value of image score
        @param: process_count - Count of processes
        @return: PythonScriptStep
        """
        return PythonScriptStep(
            name                   = 'extract_plates',
            script_name            = 'extract_plates.py',
            source_directory       = self.source_dir,
            arguments              = [
                '--indir', input_name,
                '--outdir', output,
                '--logdir', 'outputs',
                '--min_score', min_score,
                '--process_count', process_count,
            ],
            inputs                 = [input],
            outputs                = [output],
            compute_target         = self.compute_target,
            runconfig              = self.run_config,
            allow_reuse            = self.allow_reuse,
        )

    def step_split_train_test(
        self, inputs, 
        training_data, testing_data, prev_valid=None, prev_train=None, 
        split_rate=0.9, seed=100, process_count=8):
        """ Make Pipeline step: split to training/testing Datasets
        @param: inputs - List of input Dataset, 1<=len(inputs)<=2
        @param: training_data - Output training Dataset
        @param: testing_data - Output testing Dataset
        @param: exclude - Exclude from training Dataset (prev. testing)
        @param: split_rate - Split rate, example: 0.9 - 90% training, 10% testing
        @param: seed - Random seed
        @param: process_count - Count of processes
        @return: PythonScriptStep
        """
        inputs_arg = [x for d in inputs for x in ['-d', d]]
        prev_valid_arg = ['--prev_valid', prev_valid] if prev_valid else []
        prev_train_arg = ['--prev_train', prev_train] if prev_train else []
        return PythonScriptStep(
            name                   = 'split_train_test',
            script_name            = 'split_train_test.py',
            source_directory       = self.source_dir,
            arguments              = [
                '--trainroot', training_data,
                '--valroot', testing_data,
                '--split_rate', split_rate,
                '--seed', seed,
                '--process_count', process_count
            ] + inputs_arg + prev_valid_arg + prev_train_arg,
            inputs                 = inputs + ([prev_valid] if prev_valid else []) + ([prev_train] if prev_train else []),
            outputs                = [training_data, testing_data],
            compute_target         = self.compute_target,
            runconfig              = self.run_config,
            allow_reuse            = self.allow_reuse,
        )

    def step_mk_lmdb(self, inputs, output, min_score=0.1):
        """ Make Pipeline step: make LMDB Dataset.
            NOTE: Avoid using mk_lmdb_dataset as a separate step on AzureML 
            because this step takes a lot of calculation time on cloud.
            The reason is: pathlib.Path(...).rglob("*.jpg") function.
            For example, to conver ~3M images 128x64 into LMDB it takes ~1day
        @param: inputs - List of Dataset, 1<=len(inputs)<=2
        @param: output - Output Dataset
        @param: min_score - Min plate image score
        @return: PythonScriptStep
        """
        inputs_arg = [x for d in inputs for x in ['-d', d]]
        return PythonScriptStep(
            name                   = 'mk_lmdb_dataset',
            script_name            = 'mk_lmdb_dataset.py',
            source_directory       = self.source_dir,
            arguments              = [
                '--lmdbdir', output,
                '--min_score', min_score
            ] + inputs_arg,
            inputs                 = inputs,
            outputs                = [output],
            compute_target         = self.compute_target,
            runconfig              = self.run_config,
            allow_reuse            = self.allow_reuse,
        )

    def step_augmentation(self, inputs, augs, augm_data, augm_lmdb='', stacked_coeff=1, min_score=0.1, min_br=60, seed=100, process_count=8):
        """ Make Pipeline step: make augmented Dataset
        @param: inputs - List of input datasets
        @param augs - List of augmentations per unique plate text for each of inputs
        @param: augm_data - Output augmented .jpg data
        @param: augm_lmdb - Output augmented .mdb data
        @param stacked_coeff - Factor for stacked plates.
        @param: min_score - Min plate image score to save augmentation
        @param: min_br - Min plate image brightness to save augmentation
        @param: seed - random seed
        @param: process_count - Count of processes
        @return: PythonScriptStep
        """
        inputs_arg = [x for d in inputs for x in ['-d', d]]
        augs_arg = [x for a in augs for x in ['-a', a]]
        outputs = [augm_data, augm_lmdb] if augm_lmdb else [augm_data]
        return PythonScriptStep(
            name                   = 'augmentation',
            script_name            = 'augmentation.py',
            source_directory       = self.source_dir,
            arguments              = [
                '--outdir', augm_data,
                '--logdir', 'outputs',
                '--stacked_coeff', stacked_coeff,
                '--min_score', min_score,
                '--min_br', min_br,
                '--seed', seed,
                '--process_count', process_count,
            ] + (['--out_lmdb', augm_lmdb] if augm_lmdb else []) + inputs_arg + augs_arg,
            inputs                 = inputs,
            outputs                = outputs,
            compute_target         = self.compute_target,
            runconfig              = self.run_config,
            allow_reuse            = self.allow_reuse,
        )

    def run_pipeline(self, steps, experiment_name):
        """ Submit pipeline and wait for completion
        @param: steps - Pipeline steps
        @param: experiment_name - Name of experiment
        @return: None
        """
        pipeline = Pipeline(
            workspace = self.workspace,
            steps = steps
        )
        run = pipeline.submit(experiment_name=experiment_name)
        run.wait_for_completion()

class ALPRTrain(ALPRExperiment):
    """ Class of ALPR training experiment.
    """
    def __init__(self, cluster_name, conda_cfg, source_dir, env_name, is_register_env=False, curated_env_name=None, 
        env_vars={}, pip_pkgs=[], conda_pkgs=[], datastore_name=None):
        """
        @param: cluster_name - Cluster name
        @param: conda_cfg - Conda config .yml file
        @param: source_dir - Dir. of training script
        @param: env_name - Env. name (if it not exists, then create a new one)
        @param: is_register_env - If need to register new env.
        @param: curated_env_name - AzureML curated env. as a base env. for new env.
        @param: env_vars - Env. variables
        @param: pip_pkgs - Pip packages
        @param: conda_pkgs - Conda packages
        @param: datastore_name - Datastore name
        """
        super(ALPRTrain, self).__init__(cluster_name, source_dir, datastore_name)
        
        self.environment = get_environment(
            self.workspace, env_name, is_register_env, curated_env_name, 
            conda_cfg, env_vars, pip_pkgs, conda_pkgs)

    def run(self, experiment_name, model_name, trainroot, valroot, as_mount=True, parent_model='', 
        script_params=[], tags={}, node_count=4, process_count=8, tag=''):
        """ Run of training experimsnt
        @param: experiment_name - Name of experiment
        @param: model_name - Model name: GlamdringV10, ...
        @param: trainroot - Training dataset 'DATASET:VERSION'
        @param: valroot - Validation dataset 'DATASET:VERSION'
        @param: as_mount - True: as_mount, False: as download
        @param: parent_model - Pre-trained model 'NAME:VERSION'
        @param: script_params - Addition script parameters
        @param: tags - Experiment tags and for model registration
        @param: node_count - Node count
        @param: process_count - Count of processes
        @param: tag - Tag to mark output files: model and logs
        @return: None - wait for completion of experiment 
        """

        train_dataset = self.source_dataset(trainroot, as_mount)
        valid_dataset = self.source_dataset(valroot,  as_mount)

        src = ScriptRunConfig(
            source_directory = self.source_dir,
            script           = 'train.py',
            arguments = [
                '--tag',           tag,
                '--cfg',           model_name,
                '--trainroot',     train_dataset,
                '--valroot',       valid_dataset,
                '--model_path',    parent_model,
            ] + script_params,
            compute_target   = self.compute_target,
            environment      = self.environment,
            # NOTE: Don't use PyTorchConfiguration(process_count = ..., node_count = ...)
            distributed_job_config = MpiConfiguration(
                process_count_per_node = process_count,
                node_count             = node_count
            )
        )
        run = Experiment(self.workspace, experiment_name).submit(
            src, 
            tags = tags)
        run.wait_for_completion(show_output=True)

    def run1(self, experiment_name, model_name, train_dataset, test_dataset, pretrained_model='', 
        script_params=[], tags={}, node_count=4, process_count=8, tag=''):
        """ Deprecated version
        """
        src = ScriptRunConfig(
            source_directory = self.source_dir,
            script           = 'main.py',
            arguments = [
                '--tag',           tag,
                '--cfg',           model_name,
                '--trainroot',     train_dataset,
                '--valroot',       test_dataset,
                '--model_path',    pretrained_model,
            ] + script_params,
            compute_target   = self.compute_target,
            environment      = self.environment,
            # NOTE: Don't use PyTorchConfiguration(process_count = ..., node_count = ...)
            distributed_job_config = MpiConfiguration(
                process_count_per_node = process_count,
                node_count             = node_count
            )
        )
        run = Experiment(self.workspace, experiment_name).submit(
            src, 
            tags = tags)
        run.wait_for_completion(show_output=True)

class ALPRTest(ALPRExperiment):
    """ Class of ALPR testing experiment.
    """
    def __init__(self, cluster_name, conda_cfg, source_dir, env_name, is_register_env=False, curated_env_name=None, 
        env_vars={}, pip_pkgs=[], conda_pkgs=[], datastore_name=None):
        """
        @param: cluster_name - Cluster name
        @param: conda_cfg - Conda config .yml file
        @param: source_dir - Dir. of training script
        @param: env_name - Env. name (if it not exists, then create a new one)
        @param: is_register_env - If need to register new env.
        @param: curated_env_name - AzureML curated env. as a base env. for new env.
        @param: env_vars - Env. variables
        @param: pip_pkgs - Pip packages
        @param: conda_pkgs - Conda packages
        @param: datastore_name - Datastore name
        """
        super(ALPRTest, self).__init__(cluster_name, source_dir, datastore_name)
        
        self.environment = get_environment(
            self.workspace, env_name, is_register_env, curated_env_name, 
            conda_cfg, env_vars, pip_pkgs, conda_pkgs)
    
    def run(self, experiment_name, model_name, models, datasets, as_mount=True,
        script_params=[], tags={}, node_count=4, process_count=8, tag=''):
        """Run of esting models on datasets.
        @param: experiment_name - Name of experiment
        @param: model_name - Model name: GlamdringV10, ...
        @param: models - Models: ['NAME1:VERSION1', 'NAME2:VERSION2', ..]
        @param: datasets - Datasets: ['NAME1:VERSION1', 'NAME2:VERSION2', ..]
        @param: as_mount - True: as mount, False: as download
        @param: script_params - Addition script parameters
        @param: tags - Experiment tags
        @param: node_count - Node count
        @param: process_count - Count of processes
        @param: tag - Tag to mark output files: logs
        @return: None - wait for completion of experiment 
        """
        datasets = [self.source_dataset(d, as_mount) for d in datasets]
        src = ScriptRunConfig(
            source_directory = self.source_dir,
            script           = 'test.py',
            arguments = [
                '--tag', tag,
                '--cfg', model_name,
            ] + [x for m in models for x in ['-m', m]] + [x for d in datasets for x in ['-d', d]] + script_params,
            compute_target   = self.compute_target,
            environment      = self.environment,
            # NOTE: Don't use PyTorchConfiguration(process_count = ..., node_count = ...)
            distributed_job_config = MpiConfiguration(
                process_count_per_node = process_count,
                node_count             = node_count
            )
        )
        run = Experiment(self.workspace, experiment_name).submit(
            src, 
            tags = tags)
        run.wait_for_completion(show_output=True)

    def run1(self, experiment_name, model_name, models, datasets, as_mount=True,
        script_params=[], tags={}, node_count=4, process_count=8, tag=''):
        """ Deprecated version
        """
        datasets = [self.source_dataset(d, as_mount) for d in datasets]
        src = ScriptRunConfig(
            source_directory = self.source_dir,
            script           = 'main.py',
            arguments = [
                '--cfg', model_name,
                '--tag', tag,
                '--azureml', 'True',
                '--valroot', datasets[0],
                '--model_path', models[0],
            ] + script_params,
            compute_target   = self.compute_target,
            environment      = self.environment,
            # NOTE: Don't use PyTorchConfiguration(process_count = ..., node_count = ...)
            distributed_job_config = MpiConfiguration(
                process_count_per_node = process_count,
                node_count             = node_count
            )
        )
        run = Experiment(self.workspace, experiment_name).submit(
            src, 
            tags = tags)
        run.wait_for_completion(show_output=True)