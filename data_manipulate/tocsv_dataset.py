import tensorflow_datasets as tfds
import pandas as pd

def view_dataset(dataset_name):
    return tfds.load(dataset_name, split=['test'], download=False, data_dir=f'./db/dataset/{dataset_name}', with_info=True)

def to_data_frame_cifar100():
    ds, info = view_dataset('cifar100')
    label_names = info.features['label'].names
    coarse_label_names = info.features['coarse_label'].names

    data_frame = []

    for item in ds:
        data_frame.append({
            'image': item['image'].numpy(),
            'coarse_label': item['coarse_label'].numpy(),
            'label': item['label'].numpy(),
            'coarse_label_name': coarse_label_names[item['coarse_label']],
            'label_name': label_names[item['label']]
        })

    df = pd.DataFrame(data_frame)
    df.to_csv('./db/data_frames/cifar100/cifar100_full.csv', sep=',', index=False)

to_data_frame_cifar100()