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

"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensor2tensor import common_layers


def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.

  Args:
    memory_padding: a boolean `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = tf.to_float(memory_padding) * -1e9
  return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def attention_image_summary(attn, image_shapes=None):
  """Compute color image summary.

  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional quadruple of integer scalars.
      If the query positions and memory positions represent the
      pixels of a flattened image, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
  """
  num_heads = attn.get_shape().as_list()[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  if image_shapes is not None:
    q_rows, q_cols, m_rows, m_cols = list(image_shapes)
    image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
    image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
    image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
  tf.summary.image("attention", image, max_outputs=1)


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
      If the query positions and memory positions represent the
      pixels of a flattened image, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope( name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if summaries and not tf.get_variable_scope().reuse:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        summaries=False,
                        image_shapes=None,
                        name=None,
                        reuse=False):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
      If the query positions and memory positions represent the
      pixels of a flattened image, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="multihead_attention", reuse=reuse):

    # self attention
    combined = common_layers.conv1d(query_antecedent, total_key_depth * 2 + total_value_depth, 1, name="qkv_transform")
    q, k, v = tf.split(combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    x = dot_product_attention(
        q, k, v, bias, dropout_rate, summaries, image_shapes)
    x = combine_heads(x)
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    return x
