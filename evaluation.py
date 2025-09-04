# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing FID/Inception scores."""

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.keras.applications.inception_v3 import preprocess_input

# INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
# INCEPTION_OUTPUT = 'logits'
# INCEPTION_FINAL_POOL = 'pool_3'
# _DEFAULT_DTYPES = {
#     INCEPTION_OUTPUT: tf.float32,
#     INCEPTION_FINAL_POOL: tf.float32
# }
# INCEPTION_DEFAULT_IMAGE_SIZE = 299

def get_inception_model(inceptionv3=False):
    return tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

def load_dataset_stats(config):
    """Load the pre-computed dataset statistics."""
    if config.dataset == 'cifar10':
        filename = 'assets/stats/cifar10_stats.npz'
    else:
        raise ValueError(
            'No FID statistics for %s are available' % config.dataset)

    with tf.io.gfile.GFile(filename, 'rb') as fin:
        stats = np.load(fin)
        return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
    """Returns a function that can be as a classifier function.

    Copied from tfgan but avoid loading the model each time calling _classifier_fn

    Args:
      output_fields: A string, list, or `None`. If present, assume the module
        outputs a dictionary, and select this field.
      inception_model: A model loaded from TFHub.
      return_tensor: If `True`, return a single tensor instead of a dictionary.

    Returns:
      A one-argument function that takes an image Tensor and returns outputs.
    """
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]

    def _classifier_fn(images):
        output = inception_model(images)
        if output_fields is not None:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output = list(output.values())[0]
        return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

    return _classifier_fn

def _compute_global_batch_size(n: tf.Tensor, num_batches: int) -> tf.Tensor:
    num_batches = tf.maximum(1, tf.cast(num_batches, tf.int32))
    return tf.maximum(tf.constant(1, tf.int64), n // num_batches)

@tf.function
def run_inception_jit(inputs,
                      inception_model: tf.keras.Model,
                      num_batches: int = 1,
                      inceptionv3: bool = True):  # kept for API parity; ignored
    """Runs a Keras InceptionV3 model. `inputs` assumed in [0, 255]."""
    # Preprocess for keras.applications.InceptionV3 (expects [-1, 1])
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)  # safe for images in [0, 255]

    # Build dataset with ~num_batches global batches
    n = tf.shape(x)[0]
    global_bs = _compute_global_batch_size(n, num_batches)
    ds = tf.data.Dataset.from_tensor_slices(x).batch(global_bs, drop_remainder=False)

    # Accumulate outputs in graph mode
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    i = tf.constant(0)
    for batch in ds:
        y = inception_model(batch, training=False)  # (B, 2048) if pooling='avg'
        ta = ta.write(i, y)
        i += 1

    return tf.concat(ta.stack(), axis=0)

@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
    """Distribute the inception network computation to all available TPUs.

    Args:
      input_tensor: The input images. Assumed to be within [0, 255].
      inception_model: The inception network model obtained from `tfhub`.
      num_batches: The number of batches used for dividing the input.
      inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

    Returns:
      A dictionary with key `pool_3` and `logits`, representing the pool_3 and
        logits of the inception network respectively.
    """

    
    num_tpus = jax.local_device_count()
    input_tensors = tf.split(input_tensor, num_tpus, axis=0)
    pool3 = []
    logits = [] if not inceptionv3 else None
    device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
    for i, tensor in enumerate(input_tensors):
        with tf.device(device_format.format(i)):
            tensor_on_device = tf.identity(tensor)
            res = run_inception_jit(
                tensor_on_device, inception_model, num_batches=num_batches,
                inceptionv3=inceptionv3)

            if not inceptionv3:
                pool3.append(res['pool_3'])
                logits.append(res['logits'])  # pytype: disable=attribute-error
            else:
                pool3.append(res)

    with tf.device('/CPU'):
        return {
            'pool_3': tf.concat(pool3, axis=0),
            'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
        }