import tensorflow_datasets as tfds
from module.data_pre import apply_normalize_on_dataset, normalize_and_resize_img

# 스탠포드 개 데이터셋 다운로드
(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
)

# 데이터셋 전처리&배치처리
ds_train_norm = apply_normalize_on_dataset(ds_train)
ds_val_norm = apply_normalize_on_dataset(ds_test)
num_classes = ds_info.features["label"].num_classes