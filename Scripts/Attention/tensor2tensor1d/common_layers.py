# Copyright 2017 Google Inc.
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

"""Layers common to multiple models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor import expert_utils as eu

import tensorflow as tf

from tensorflow.python.framework import function

# This is a global setting. When turned off, no @function.Defun is used.
allow_defun = True


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4.")
  inputs.set_shape([static_shape[0], None, None, static_shape[3]])
  # Add support for left padding.
  if "padding" in kwargs and kwargs["padding"] == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    kwargs["padding"] = "VALID"
  force2d = False  # Special argument we use to force 2d kernels (see below).
  if "force2d" in kwargs:
    force2d = kwargs["force2d"]

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    if "name" in kwargs:
      original_name = kwargs["name"]
      name = kwargs.pop("name") + "_" + name_suffix
    else:
      original_name = None
      name = "conv_" + name_suffix
    original_force2d = None
    if "force2d" in kwargs:
      original_force2d = kwargs.pop("force2d")
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  # Manually setting the shape to be unknown in the middle two dimensions so
  # that the `tf.cond` below won't throw an error based on the convolution
  # kernels being too large for the data.
  inputs._shape = tf.TensorShape([static_shape[0], None, None, static_shape[3]])  # pylint: disable=protected-access
  if kernel_size[1] == 1 or force2d:
    # Avoiding the cond below can speed up graph and gradient construction.
    return conv2d_kernel(kernel_size, "single")
  return tf.cond(
      tf.equal(tf.shape(inputs)[2],
               1), lambda: conv2d_kernel((kernel_size[0], 1), "small"),
      lambda: conv2d_kernel(kernel_size, "std"))


def conv(inputs, filters, kernel_size, **kwargs):
  return conv_internal(tf.layers.conv2d, inputs, filters, kernel_size, **kwargs)


def conv1d(inputs, filters, kernel_size, **kwargs):
  return tf.squeeze(
      conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1), **kwargs), 2)

