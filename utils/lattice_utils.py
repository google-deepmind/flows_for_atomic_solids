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

"""Utilities to build lattice-based particle models."""

from typing import Tuple, Union
import chex
import numpy as np

Array = chex.Array


def gcd(values: Array, tol: float = 3e-2) -> Union[int, float]:
  """GCD of a list of numbers (possibly floats)."""
  def _gcd2(a, b):
    if np.abs(b) < tol:
      return a
    else:
      return _gcd2(b, a % b)
  x = values[0]
  for v in values[1:]:
    x = _gcd2(x, v)
  return x


def make_simple_lattice(lower: Array,
                        upper: Array,
                        cell_aspect: Array,
                        n: int) -> Tuple[Array, Array, Array]:
  """Returns a shifted cubic lattice with atoms at the unit cell centres."""
  dim = len(lower)
  assert len(upper) == dim
  assert len(cell_aspect) == dim
  box_size = upper - lower
  normalized_box = (upper - lower) / cell_aspect
  integer_aspect = np.round(normalized_box / gcd(normalized_box)).astype(int)
  num_per_dim = np.round((n/np.prod(integer_aspect)) ** (1/dim)).astype(int)
  repeats = num_per_dim * integer_aspect
  if np.prod(repeats) != n:
    raise ValueError(f'The number of lattice points {n} does not '
                     f'match the box size {box_size} and cell aspect '
                     f'{cell_aspect}, got integer aspect {integer_aspect}, '
                     f'{repeats} repeats.')
  points = [np.linspace(lower[i], upper[i], repeats[i], endpoint=False).T
            for i in range(dim)]
  xs = np.meshgrid(*points)
  lattice = np.concatenate([x[..., None] for x in xs], axis=-1)
  lattice = lattice.reshape(np.prod(repeats), dim)
  lattice_constant = (upper - lower) / repeats

  return lattice, lattice_constant, repeats


def make_lattice(lower: Array,
                 upper: Array,
                 cell_aspect: Array,
                 atom_positions_in_cell: Array,
                 n: int) -> Tuple[Array, Array, Array]:
  """An orthorhombic lattice of repeated unit cells.

  Args:
    lower: vector of lower limits of lattice box (number of elements
      determines the dimensionality)
    upper: vector of upper limits of lattice box (number of elements
      must be the same as `lower`)
    cell_aspect: relative lengths of unit cell edges. A cubic cell would have
      `cell_aspect==[1, 1, 1]` for example. The box basis `lower - upper`,
      divided by `cell_aspect`, should have low-integer length ratios.
    atom_positions_in_cell: a n_u x dimensionality matrix with the fractional
      positions of atoms within each unit cell. `n_u` will be the number
      of atoms per unit cell.
    n: number of atoms in lattice. Should be the product of the low-integer
      length ratios of the aspect-normalized box, times some integer to the
      power of the number of dimensions, times the number of atoms per
      cell.

  Returns:
    A 3-tuple (lattice, lattice_constant, repeats):
      lattice: n x dimension array of lattice sites.
      lattice_constant: a vector of length equal to the dimensionality, with
        the side lengths of the unit cell.
      repeats: an integer vector of length equal to the dimensionality, with
        the number of cells in each dimension. `repeats x lattice_constant`
        equals `upper - lower`.
  """
  num_cells = n // len(atom_positions_in_cell)
  if num_cells * len(atom_positions_in_cell) != n:
    raise ValueError(f'Number of particles {n} is not divisible by the number '
                     f'of particles per cell {len(atom_positions_in_cell)}')
  base, lc, repeats = make_simple_lattice(lower, upper, cell_aspect, num_cells)
  sites = atom_positions_in_cell * lc
  lattice = base[..., None, :] + sites
  lattice = lattice.reshape(-1, lattice.shape[-1])
  return lattice, lc, repeats
