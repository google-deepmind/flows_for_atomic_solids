#!/usr/bin/python
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention modules."""

import math
from typing import Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

Array = chex.Array


class Attention(hk.Module):
  """Multi-headed dot-product attention."""

  def __init__(self,
               num_heads: int,
               w_init: Optional[hk.initializers.Initializer] = None,
               w_init_final: Optional[hk.initializers.Initializer] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_heads = num_heads
    default = hk.initializers.VarianceScaling(1.)
    self._w_init = default if w_init is None else w_init
    self._w_init_final = default if w_init_final is None else w_init_final

  @hk.transparent
  def _multihead_linear(self, x: Array, head_size: int) -> Array:
    """Linearly project `x` to have `head_size` dimensions per head."""
    out = hk.Linear(self._num_heads * head_size, w_init=self._w_init)(x)
    shape = out.shape[:-1] + (self._num_heads, head_size)
    return jnp.reshape(out, shape)

  def __call__(self, q: Array, k: Array, v: Array) -> Array:
    """Apply attention with queries `q`, keys `k` and values `v`.

    Args:
      q: array of shape [..., N_q, D_q].
      k: array of shape [..., N_k, D_k].
      v: array of shape [..., N_k, D_v].

    Returns:
      Array of shape [..., N_q, D_q].
    """
    num_dims = q.shape[-1]
    if num_dims % self._num_heads != 0:
      raise ValueError(f'The number of dimensions ({num_dims}) is not divisible'
                       f' by the number of heads ({self._num_heads}).')
    head_size = num_dims // self._num_heads
    # Preprocess queries, keys and values.
    q = self._multihead_linear(q, head_size)
    k = self._multihead_linear(k, head_size)
    v = self._multihead_linear(v, head_size)
    # Compute attention matrix.
    scale = math.sqrt(head_size)
    attention = jnp.einsum('...thd,...Thd->...htT', q, k) / scale
    attention = jax.nn.softmax(attention)
    # Attend over values and concatenate head outputs.
    attended_v = jnp.einsum('...htT,...Thd->...thd', attention, v)
    attended_v = jnp.reshape(attended_v, attended_v.shape[:-2] + (num_dims,))
    return hk.Linear(num_dims, w_init=self._w_init_final)(attended_v)


class SelfAttention(Attention):
  """Multi-headed dot-product self-attention."""

  def __call__(self, x: Array) -> Array:
    return super().__call__(x, x, x)


class _DenseBlock(hk.Module):
  """An MLP with one hidden layer, whose output has the size of the input."""

  def __init__(self,
               widening_factor: int,
               w_init: hk.initializers.Initializer,
               w_init_final: hk.initializers.Initializer,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._w_init = w_init
    self._w_init_final = w_init_final

  def __call__(self, x: Array) -> Array:
    num_dims = x.shape[-1]
    num_hiddens = self._widening_factor * num_dims
    x = hk.Linear(num_hiddens, w_init=self._w_init)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(num_dims, w_init=self._w_init_final)(x)


def _layer_norm(x: Array, name: Optional[str] = None) -> Array:
  """Apply a unique LayerNorm to `x` with default settings."""
  return hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,
                      name=name)(x)


class Transformer(hk.Module):
  """A transformer model."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               dropout_rate: float = 0.,
               use_layernorm: bool = True,
               w_init: Optional[hk.initializers.Initializer] = None,
               w_init_final: Optional[hk.initializers.Initializer] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._dropout_rate = dropout_rate
    if use_layernorm:
      self._maybe_layer_norm = _layer_norm
    else:
      self._maybe_layer_norm = lambda h, name: h
    default = hk.initializers.VarianceScaling(2. / math.sqrt(num_layers))
    self._w_init = default if w_init is None else w_init
    self._w_init_final = default if w_init_final is None else w_init_final

  def __call__(self, x: Array, is_training: Optional[bool] = None) -> Array:
    """Applies the transformer.

    Args:
      x: array of shape [..., num_points, num_dims].
      is_training: whether we're training or not. Must be provided when dropout
        is used, otherwise it can be left unspecified.

    Returns:
      Array of same shape as `x`.
    """
    if self._dropout_rate != 0. and is_training is None:
      raise ValueError('`is_training` must be specified when dropout is used.')
    dropout_rate = self._dropout_rate if is_training else 0.
    h = x
    for i in range(self._num_layers):
      h_norm = self._maybe_layer_norm(h, name=f'h{i}_ln_1')
      h_attn = SelfAttention(num_heads=self._num_heads,
                             w_init=self._w_init,
                             w_init_final=self._w_init_final,
                             name=f'h{i}_attn')(h_norm)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = self._maybe_layer_norm(h, name=f'h{i}_ln_2')
      h_dense = _DenseBlock(widening_factor=4,
                            w_init=self._w_init,
                            w_init_final=self._w_init_final,
                            name=f'h{i}_mlp')(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    return self._maybe_layer_norm(h, name='ln_f')
