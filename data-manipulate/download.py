import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def download_tf_datasets(dataset_name):
    print(f'Baixando Dataset do TenforFlow: {dataset_name}...')
    ds = tfds.load(dataset_name, split=['train', 'test'], shuffle_files=True, with_info=True, download=True, data_dir=f'./db/dataset/{dataset_name}')
    print(f"Download do dataset: {dataset_name} Concluido")

download_tf_datasets('cifar100')