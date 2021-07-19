# ALPR OCR


## Training

```shell
python3 train.py --cfg 'MODEL_NAME' [--model_path 'PRETRAINED_MODEL'] --trainroot 'TRAIN_DIR' --valroot 'TEST_DIR' --outdir 'OUTPUT_DIR' --batch_size '1024' --lr '1e-4' --lr_period '10' --min_lr '1e-7' --epochs '1000' --seed '42'  --tag 'TAG'
```

## Testing

Testing multiple models on multiple datasets
```shell
sudo python3 test.py --cfg 'MODEL_NAME' -m 'MODEL_PATH1' [-m 'MODEL_PATH2' ...] -d 'DATASET_PATH1' [-d 'DATASET_PATH2' ...] --outdir 'OUTPUT_DIR' [--error_folder 'ERROR_DIR']  --tag 'TAG'
```

## Inferring 

```
python infer.py --model_path models/CRNNBiris/CRNNBiris_H64_W128_S128_265_train_clean.pth --img_path images/VAN\ 610.jpg 
```

## Training on AzureML

Go to: [azureml_jobs](azureml_jobs)
