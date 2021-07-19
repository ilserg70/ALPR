# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from time import time
from pathlib import Path

from azureml.core.run import Run
from azureml.core import Dataset
from azureml.core.datastore import Datastore

import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--names', type=str, required=True, help="Input data names (comma sep.). Use: inputs=[data.as_named_input(name)]")
    parser.add_argument('--dscrs', type=str, required=True, help="Descriptions (comma sep.).")
    parser.add_argument('--tags',  type=str, default='',    help="Tags: 'TAG1:VAL1,TAG2:VAL2,...'")
    parser.add_argument('--datastore_name', type=str, default='', help='Name of datastore. Please refer to AzureML dashboard')
    args = parser.parse_args()

    print(f"Start registration dataset(s) on AzureML. Time: {str(datetime.now())}")
    start_time = time()
    utils.print_args(args)

    exp_run = Run.get_context() # Current experiment Run

    if 'OfflineRun' in exp_run._run_id:
        print("OfflineRun. No registration possible.")
    else:
        exp_name   = exp_run._experiment._name
        run_number = exp_run._run_number
        workspace  = exp_run._experiment._workspace
        datastore  = Datastore.get(workspace, args.datastore_name) if args.datastore_name else workspace.get_default_datastore()

        names = args.names.split(',')
        dscrs = args.dscrs.split(',')
        assert len(names)==len(dscrs), "Error! Number of names and descriptions are not the same."

        tags = dict([tuple(kv.split(':')) for kv in args.tags.split(',') if ':' in kv])
        tags['experiment'] = f"{exp_name} Run {run_number}"
        for name, dscr in zip(names, dscrs):
            print(name)
            source = exp_run.input_datasets[name]
            num_samples = utils.get_num_samples(source)
            dscr += f" Samples: {num_samples}"
            print(dscr)
            if num_samples>0:
                print(f"Start uploading files to datastore ...")
                datastore.upload(src_dir=source, target_path=name, overwrite=True, show_progress=True)
                print(f"Creating dataset {name} ...")
                ds = Dataset.File.from_files(path=[(datastore, name)])
                print(f"Registering dataset {name} ...")
                tags_ = tags.copy()
                tags_['num_samples'] = num_samples
                ds.register(workspace, name, description=dscr, tags=tags_, create_new_version=True)
            else:
                print(f"Faled registration dataset: {name} - No data.")

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
