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

"""Utilities for running experiments."""

from typing import Callable, Optional, Sequence, Union
import chex
import jax.numpy as jnp
import numpy as np

Array = chex.Array
Numeric = Union[Array, int, float]


def get_lr_schedule(base_lr: float,
                    lr_decay_steps: Sequence[int],
                    lr_decay_factor: float) -> Callable[[Numeric], Numeric]:
  """Returns a callable that defines the learning rate for a given step."""
  if not lr_decay_steps:
    return lambda _: base_lr

  lr_decay_steps = jnp.array(lr_decay_steps)
  if not jnp.all(lr_decay_steps[1:] > lr_decay_steps[:-1]):
    raise ValueError('Expected learning rate decay steps to be increasing, got '
                     f'{lr_decay_steps}.')

  def lr_schedule(update_step: Numeric) -> Array:
    i = jnp.sum(lr_decay_steps <= update_step)
    return base_lr * lr_decay_factor**i

  return lr_schedule


def get_orthorhombic_box_lengths(
    num_particles: int, density: float, dim: int, shape_factor: Array,
    repeats: Optional[Array]) -> Array:
  """Returns edge lengths of an orthorhombic box."""
  assert dim == len(shape_factor)
  vol = num_particles / density
  if repeats is None:
    repeats = np.ones(dim, dtype=int)
  base = (vol / np.prod(shape_factor * repeats)) ** (1./dim)
  return base * shape_factor * repeats


def get_hexagonal_box_lengths(
    num_particles: int, density: float, dim: int,
    repeats: Optional[Array] = None) -> Array:
  """Returns edge lengths of an orthorhombic box for Ih packing."""
  shape_factor = np.array([1.0, np.sqrt(3), np.sqrt(8/3)])
  return get_orthorhombic_box_lengths(
      num_particles, density, dim, shape_factor, repeats)


def get_cubic_box_lengths(
    num_particles: int, density: float, dim: int) -> Array:
  """Returns the edge lengths of a cubic simulation box."""
  edge_length = (num_particles / density) ** (1./dim)
  return np.full([dim], edge_length)
