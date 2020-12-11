# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'f29420a1-6637-4252-bc28-67131742f1f7'
resource_group = 'BatchAI-sandbox'
workspace_name = 'QUS5UW2-ML-Sandbox'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='ALPR_video_based_Evaluation_test_CALIFORNIA_12_files')
dataset.download(target_path='./data/ALPR_video_based_Evaluation_test_CALIFORNIA_12_files/', overwrite=False)