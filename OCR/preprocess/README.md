# Data preprocessing

## Extract plates
[algorithm](https://axon.quip.com/63f5A1O2dggC/ALPR-OCR-data-preparation-algorithm)
This is the first step of data preparation for ALPR OCR training - extraction of plate images:
- Extraction of plate sub-images from frame images using annotations.
- Calculation of `score` for plate images.
- Cleaning of plate text.
- Checking plate text annotation validity.
- Take the first char from all blocks of uncertainty (angle brackets). Examples: 
    - `<OQ>RT556`    -> `ORT556`
    - `A<EFP>X888`   -> `AEX888`
    - `334<8B>W<TY>` -> `3348WT`
- Converting US States to canonical names.
- Restoring missed US States using grouping plates by text.

```shell
python3 extract_plates.py --indir 'JPG_INPUT' --outdir 'JPG_OUTPUT' --logdir 'LOG_OUTPUT'
```

## Split into training/testing datasets
Split plate images (`.jpg`) into training and testing datasets
- Input files may contain many duplicates - different images of the same licence plate (from different video frames). 
- The point is to split data into non overlapped 'train' and 'test' datasets - duplicates belong to either 'train' or 'test' subset.

```shell
python3 split_train_test.py -d 'JPG_INPUT1' [-d 'JPG_INPUT2' ...]  --trainroot 'JPG_TRAIN' --valroot 'JPG_TEST' [--prev_train 'DATASET:VERSION'] [--prev_valid 'DATASET:VERSION'] --split_rate 0.9 --seed 42
```

## Inspect dataset(s)
Get stat info about dataset and calc. of datasets intersection - how many common licence plates (from different frames). Datasets: .jpg or .mdb or mix.

```shell
python3 preprocess/inspect_dataset.py -d 'DIR1' [-d 'DIR2' ...] --logdir 'LOGDIR'
```

```shell
sudo python3 inspect_dataset.py -d '/media/6TB/alpr_data/ALPR_OCR_PLv7_2_test' -d '/media/6TB/alpr_data/ALPR_OCR_PLv7_2_train' -d '/media/6TB/alpr_data/ALPR_OCR_PLv7_3_test' -d '/media/6TB/alpr_data/ALPR_OCR_PLv7_3_train' --logdir '/home/silinskiy/mydata/tmp/7_2_vs_7_3'
```

## Augmentation
Enriching data (`.jpg` files) with a new images by transformation of existing images with OpenCV transformations.
- [Description](../azureml_jobs/README.md)
- [Examples](https://axon.quip.com/UzJ4AgIEcwpQ/ALPR-OCR-Data-augmentation)
- [OpenCV description](https://docs.opencv.org/master/)
- [Image transformations](image_transformation.py) - is a set of image transformations

```shell
python3 augmentation.py  -d 'DIR1' [-d 'DIR2' ...] -a NUM1 [-a NUM2 ...] --stacked_coeff NUM --outdir 'OUT_DIR' [--out_lmdb 'OUT_LMDB'] --logdir 'LOG_DIR' --min_score 0.1 --seed 42
```

## Make LMDB dataset
Creation of LMDB dictionary type dataset with images as binary objects.
_*Note:*_ Avoid using mk_lmdb_dataset as a separate step on AzureML because this step takes a lot of calculation time on cloud.
          The reason is: pathlib.Path(...).rglob("*.jpg") function which takes a huge amount of time on AzureML.
          For example, to conver ~3M images 128x64 into LMDB it takes ~1day

```shell
python3 mk_lmdb_dataset.py -d 'DIR1' [-d 'DIR2' ...] --lmdbdir 'LMDB_OUTDIR' --min_score 0.1
```

## Merging datasets

```shell
python3 merge.py -d 'DIR1' [-d 'DIR2' ...] --outdir 'OUTDIR'
```

## US States similarity
Calculation of US States similarity based on plate texts

```shell
sudo python3 states_similarity.py -d 'DIR1' [-d 'DIR2' ...] --outdir 'OUT_DIR' --clear_outdir 1 --min_rate 0.9 --min_size 200
```

## Local experiment
Run experiment on local machine:
* extract_plates
* split_train_test
* augmentation
* mk_lmdb_dataset

```shell
sudo ./local_preprocess.sh 'OUT_DIR'
```

## Experiments

#### Local preprocess
```shell
ssh USER@dextro-lady-of-the-lake
cd ~/axon_git/OCR-SOR
./preprocess/local_preprocess.sh '/home/silinskiy/mydata/tmp'
```

#### Datasets

| ID   |    Dataset                                         |  Samples   |  Comment      |
|------|----------------------------------------------------|------------|---------------|
|  S0  | `/media/6TB/alpr_data/wave0/all_data_10232020/raw` |    63,589  | Fraims        |
|  S1  | `/media/6TB/alpr_data/scaleai_all`                 |   293,613  | Fraims        |
|  T0  | `/media/6TB/alpr_data/wave1_scaleai/train`         |   117,890  | Fraims        |
|  V0  | `/media/6TB/alpr_data/wave1_scaleai/valid`         |     6,729  | Fraims        |
|  US5 | `/media/6TB/alpr_data/us5_08312020`                |    74,220  | Fraims        |
|  T1  | `/media/6TB/alpr_data/wave1_hang/train_0_0219_0.0` | 3,240,300  | Plates augm.  |
|  V1  | `/media/6TB/alpr_data/wave1_hang/test_0_0219_0.0`  |    11,017  | Plates        | LMDB
|  T2  | `/media/6TB/alpr_data/ocr_PLv5_train_old_and_new`  | 2,916,405  | Plates augm.  |
|  V2  | `/media/6TB/alpr_data/ocr_PLv5_valid_old_and_new`  |    15,317  | Plates        | LMDB
|  T3  | `/home/silinskiy/mydata/tmp/train`                 |   200,429  | Plates        |
|  V3  | `/home/silinskiy/mydata/tmp/valid`                 |    19,249  | Plates        |


```shell
sudo python3 inspect_dataset.py -d '/media/6TB/alpr_data/ocr_PLv5_valid_old_and_new' -d '/media/6TB/alpr_data/ocr_PLv5_train_old_and_new' -d '/media/6TB/alpr_data/ALPR_OCR_PLv7_2_test_lmdb' -d '/media/6TB/alpr_data/ALPR_OCR_PLv7_2_train_lmdb' -d '/media/6TB/alpr_data/us5_08312020_extracted' --logdir '/media/6TB/alpr_data//intersect_datasets'
```
