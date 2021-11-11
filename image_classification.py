import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import densenet
from tensorflow.keras.utils import img_to_array


def load_densenet(name="densenet121"):
    """ Helper function to download and load densenet"""

    model = tf.keras.applications.DenseNet121(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    model.save(name)

    return model


def process_image_for_tf(image_stream):
    """This function takes a PIL image and returns a Tensor ready for Densenet121"""
    image = Image.open(image_stream)
    # resize to required size
    image_resized = image.resize((224, 224))

    # preprocess image appropriately for densenet
    img_tf = img_to_array(image_resized)
    img_tf = img_tf.reshape(1, 224, 224, 3)
    img_std = densenet.preprocess_input(img_tf)
    return img_std
