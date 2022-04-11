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

"""Embeddings."""

import chex
import jax.numpy as jnp

Array = chex.Array


def circular(x: Array,
             lower: float,
             upper: float,
             num_frequencies: int) -> Array:
  """Maps angles to points on the unit circle.

  The mapping is such that the interval [lower, upper] is mapped to a full
  circle starting and ending at (1, 0). For num_frequencies > 1, the mapping
  also includes higher frequencies which are multiples of 2 pi/(lower-upper)
  so that [lower, upper] wraps around the unit circle multiple times.

  Args:
    x: array of shape [..., D].
    lower: lower limit, angles equal to this will be mapped to (1, 0).
    upper: upper limit, angles equal to this will be mapped to (1, 0).
    num_frequencies: number of frequencies to consider in the embedding.

  Returns:
    An array of shape [..., 2*num_frequencies*D].
  """
  base_frequency = 2. * jnp.pi / (upper - lower)
  frequencies = base_frequency * jnp.arange(1, num_frequencies+1)
  angles = frequencies * (x[..., None] - lower)
  # Reshape from [..., D, num_frequencies] to [..., D*num_frequencies].
  angles = angles.reshape(x.shape[:-1] + (-1,))
  cos = jnp.cos(angles)
  sin = jnp.sin(angles)
  return jnp.concatenate([cos, sin], axis=-1)
