import tensorflow as tf
import cv2

def normalize_and_resize_img(input):
    # Normalizes images: `uint8` -> `float32`
    image = tf.image.resize(input['image'], [224, 224])
    input['image'] = tf.cast(image, tf.float32) / 255.
    return input['image'], input['label']

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=2
    )
    ds = ds.batch(batch_size)
    """
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    """
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds