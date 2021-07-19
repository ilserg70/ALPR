# AzureML jobs for ALPR OCR

This directory contains AzureML submit scripts for `ALPR OCR`:
* Data preparation - [preprocess](../preprocess)
* Training from scratch - [main.py](../main.py), [train.py](../train.py)
* Continue training (incremental learning using pre-trained model) - [main.py](../main.py), [train.py](../train.py)
* Testing [test.py](../test.py)

## Preprocess

### Plates extraction and split train/test
Extraction of plate images and creation/registeration of datasets: [algorithm](https://axon.quip.com/63f5A1O2dggC/ALPR-OCR-data-preparation-algorithm)

```shell
python3 submit_preprocess_job.py  --experiment_name 'EXP_NAME' --cluster_name 'CLUSTER' --datastore_name 'DATASTORE' --data_scaleai 'DATASET:VERSION' --data_us5 'DATASET:VERSION' [--prev_valid 'DATASET:VERSION'] [--prev_train 'DATASET:VERSION'] --PLv 'VERSION' --split_rate 0.9 --min_score 0.1
```

### Augmentation
[augmentation](https://axon.quip.com/UzJ4AgIEcwpQ/ALPR-OCR-Data-augmentation)
- We learned from experiments that training OCR model on augmented data gives increasing about ~10% accuracy.
- Wherein testing dataset contains only original samples.
- Output augmented dataset contains copies of all input images and addition augmented images.
- Augmented images are created from input images by random [transformations](../preprocess/image_transform.py):
    
    | code | weight |              function             |
    |------|--------|-----------------------------------| 
    | mot  |   2    | random_add_motion_blur            |
    | rot  |   3    | random_rotate                     |
    | r3d  |   1    | random_rotate_3D_and_warp         |
    | blr  |   1    | random_add_gaussian_blur          |
    | shd  |   1    | random_adding_shade               |
    | noi  |   1    | random_add_salt_and_pepper_noises |
    | cut  |   1    | random_cutting                    |
    | pad  |   1    | random_padding                    |
    | tbc  |   1    | random_top_bottom_cutting         |
    | brg  |   1    | random_brightness_adjustment      |
    | rsz  |   1    | random_downsize                   |

```shell
python3 submit_augmentation_job.py  --experiment_name 'EXP_NAME' --cluster_name 'CLUSTER' --datastore_name 'DATASTORE' --data_scaleai 'NAME:VERSION' --augs_scaleai NUM1 --data_us5 'NAME:VERSION' --augs_us5 NUM2 --stacked_coeff COEFF --min_score 0.1 [--trans 'mot,rot,r3d,blr,shd,noi,cut,pad,tbc,brg,rsz'] --seed 42 --PLv 'TAG'
```

## Training ALPR OCR Model
Before submit AzureML job, make sure that config files contains correct parameters:
* `--train_cfg` - [training.ini](config/training.ini) - to define training parameters which are used in the script: [main.py](../main.py), [train.py](../train.py)
* `--conda_cfg` - (Optional) [training.yml](config/training.yml) - environment. Only if need to modify `curated_name` in [submit_training_job.py](submit_training_job.py)

To run AzureML job use one of scripts:
- [submit_training_job.py](submit_training_job.py) - New version [ScriptRunConfig](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch) with pytorch & horovod
- [submit_horovod_job.py](submit_horovod_job.py) - Old version with deprecated [PyTorch estimator](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.dnn.pytorch?view=azure-ml-py)

_*NOTE!*_ Run the following commands with the --help option first to get updated version of all options.

```shell
python3 submit_training_job.py  --experiment_name 'EXP_NAME' --cluster_name 'AML_CLUSTER' --cfg 'MODEL_NAME' [--model 'PRETRAINED_MODEL:VERSION'] --trainroot 'DATASET:VERSION' --valroot 'DATASET:VERSION' --tag 'TAG'
```

Old version (with deprecated AzureML sdk)
```shell
python3 submit_horovod_job.py <... the same as above ...>
```

### NaN loss issue at training

Train function contains: `assert(~torch.isnan(cost))`. Sometimes it causes exit with loss NaN error.

#### Reason
The reason is in wrong ratio between batch size, learning rate and number of augmented duplicates per plate text.

#### What to do
To avoid this error try to use one or more from the following:
- Increase batch size
- Decrease start value of learning rate
- Use incremental learning (transfer learning with pre-trained model)

#### Example
Suppose, we have succeeded training experiment (from scratch) with `batch_size=256`, `learning_rate=1e-3` and training dataset with `augs_per_plate~10` (average number of augmented images per plate text). Now we created much bigger training dataset with `augs_per_plate~30` and experiment failed due to cost is NaN.
You can try one or more from these changes: `batch_size=1024`, `learning_rate=1e-4` and use model weights from previous experiment.

## Testing ALPR OCR Models
Testing many models on many datasets.
*  `--test_cfg` - [testing.ini](config/testing.ini) - to define testing parameters which are used in the script [test.py](../test.py)
* `--conda_cfg` - (Optional) [training.yml](config/training.yml) - the same as for training

```shell
python3 submit_testing_job.py  --experiment_name 'EXP_NAME' --cluster_name 'AML_CLUSTER' --cfg 'MODEL_NAME' -m 'MODEL1:VERSION' [-m 'MODEL2:VERSION' ...] -d 'DATASET1:VERSION' [-d 'DATASET2:VERSION' ...] --tag 'TAG'
```

## AzureML datasets

* scaleai: `scaleAI-all:1`
* US5: `us5_08312020:1`

## Examples

#### AzureML Cluster
```shell
Resource group: BatchAI-sandbox
Workspace: QUS5UW2-ML-Sandbox
Region: westus2

Standard_D64s_v3 (64 cores, 256 GB RAM, 512 GB disk)
Standard_D64_v3 (64 cores, 256 GB RAM, 1600 GB disk)
```

#### Preprocess
```shell
python3 submit_preprocess_job.py  --experiment_name 'ocr-sor-preprocess' --cluster_name 'sergey-cpu3' --data_scaleai 'scaleAI-all:1' --data_us5 'us5_08312020:1' --prev_valid 'ocr_PLv5_valid_old_and_new:1' --prev_train 'ocr_PLv5_train_old_and_new:1' --min_score 0.1 --split_rate 0.9 --process_count 16 --PLv '7_4'
```

#### Augmentation
```shell
python3 submit_augmentation_job.py  --experiment_name 'ocr-sor-augmentation' --cluster_name 'sergey-cpu3' --data_scaleai 'ALPR_OCR_PLv7_4_ScaleAI_train:1' --data_us5 'ALPR_OCR_PLv7_4_US5:1' --min_score 0.1 --augs_scaleai 20 --augs_us5 2 --stacked_coeff 1 --PLv '7_4_2'
```

#### Training
```shell
python3 submit_training_job.py  --experiment_name 'ocr-sor-hvd' --cluster_name 'gpu-sergey' --cfg 'GlamdringV10' --model 'GlamdringV10_ALPR-OCR:29' --trainroot 'ALPR_OCR_PLv7_4_2_augm_lmdb:1' --valroot 'ALPR_OCR_PLv7_4_ScaleAI_valid_lmdb:1' --node_count 4
```

#### Testing
```shell
python3 submit_testing_job.py --experiment_name 'ocr-sor-test' --cluster_name 'gpu-sergey' --cfg 'GlamdringV10' -m 'GlamdringV10_ALPR-OCR:8' -m 'GlamdringV10_ALPR-OCR:29' -d 'ALPR_OCR_PLv7_4_ScaleAI_valid_lmdb:1' --tag 'PLv5_vs_PLv7'
```

## Results

* [ALPR-OCR-PLv5](https://axon.quip.com/q3dbAdfKm4mm/ALPR-OCR-PLv5)
* [ALPR-OCR-PLv7](https://axon.quip.com/4GaRAxMiXtcb/ALPR-OCR-PLv7)
* [ALPR-PLv7-OCR-Training-Swimming-Lanes](https://axon.quip.com/yaO1AEbLHffa/ALPR-PLv7-OCR-Training-Swimming-Lanes)


ocr-sor-preprocess Run 129 (7)
    Number of extracted plates: 227,288
    Errors: 470050 (67.41%) of total 697,338

ocr-sor-preprocess Run 136 (7_2)
    Number of extracted plates: 240,043
    Errors: 490561 (67.14%) of total 730,604
    ALPR_OCR_PLv7_2_test
    ALPR_OCR_PLv7_2_train

ocr-sor-preprocess Run 145 (7_3)
    Number of extracted plates: 184,178
    Errors: 471448 (71.91%) of total 655,626
    Unique plates: 100500
    Stacked: 4857 (2.637%)
    ALPR_OCR_PLv7_3_test
    ALPR_OCR_PLv7_3_train

ocr-sor-preprocess Run 152
Number of extracted plates: 237354
Errors: 499080 (67.77%) of total 736434