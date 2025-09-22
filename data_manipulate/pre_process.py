import tensorflow_datasets as tfds
import pickle as pk

def view_dataset(dataset_name):
    return tfds.load(dataset_name, split='train', download=False, data_dir=f'./db/dataset/{dataset_name}', with_info=True)

def to_data_frame_cifar100():
    ds, info = view_dataset('cifar100')
    label_names = info.features['label'].names
    coarse_label_names = info.features['coarse_label'].names

    datas = []
    for item in ds:
        datas.append({
            'image': item['image'].numpy(),
            'coarse_label': item['coarse_label'].numpy(),
            'label': item['label'].numpy(),
            'coarse_label_name': coarse_label_names[item['coarse_label']],
            'label_name': label_names[item['label']]
        })

    for i, data in enumerate(datas):
        datas[i]['image'] = data['image'] / 255
    
    with open('./db/datas/cifar100/data_train.pkl', mode='wb') as f:
        pk.dump(datas, f)


to_data_frame_cifar100()