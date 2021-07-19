# [AzureML] Experiment and Model tags keys.
# Tags of experiment and Tags of models must be the same. So, tags modification should be done only here.

# Explanation:
# These tags are using in two places:
#    1) As experiment tags: AzureML => Experiment => Details: Tags; (submit_azureml_job.py)
#    2) As model      tags: AzureML => Model =>      Details: Tags; (azureml_utils.py, ../main.py)
# At the end of experiment but before register a new model on AzureML we check first if no model with 
# better accuracy to avoid registration of poore model. We use: ACC_TAG, TRAIN_TAG, TEST_TAG

# NOTE: After adding a new tag, add it into TAGS as well.

ACC_TAG = 'acc'
TRAIN_TAG = 'data_train'
TEST_TAG = 'data_test'
PARENT_MODEL_TAG = 'parent_model'
SCRIPT_TAG = 'script'
EXPERIMENT_TAG = 'experiment'
MSG_TAG = 'msg'

TAGS = [ACC_TAG, TRAIN_TAG, TEST_TAG, PARENT_MODEL_TAG, SCRIPT_TAG, EXPERIMENT_TAG, MSG_TAG]
