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

"""Lennard-Jones system."""

from typing import Optional

import chex
from flows_for_atomic_solids.systems import energies

import jax.numpy as jnp

Array = chex.Array


class LennardJonesEnergy(energies.PairwisePotentialEnergy):
  """Evaluates the Lennard-Jones (LJ) energy with periodic boundary conditions.

  This class implements a "soft" version of the original LJ potential and uses a
  scalar parameter `lambda_lj` in [0, 1] to interpolate between a uniform
  distribution (for `lambda_lj=0`) and the standard LJ potential (for
  `lambda_lj=1`).

  Two LJ particles, separated by a distance `r`, interact with each other via
  a radially symmetric, pairwise potential
  ```
      g(r) = 0.5 * (1 - lambda_lj)**2 + (r/sigma)**6
      u(r) = 4 * lambda_lj * epsilon * [1/g(r)**2 - 1/g(r)]
  ```
  where `epsilon` and `sigma` are two Lennard-Jones parameters defining the
  scales of energy and length.

  For `lambda_lj=1`, the pairwise potential above exhibits a singularity at
  `r=0`, so that the energy diverges whenever any two particles coincide. The
  option `min_distance` can be used to clip the pairwise distance to avoid this
  problem.

  Optionally, the energy can be shifted so that it is zero at and beyond the
  cutoff
  ```
      u_shifted(r) = u(r) - u(cutoff) if r <= cutoff and 0 otherwise.
  """

  def __init__(self,
               cutoff: float,
               box_length: Optional[Array] = None,
               epsilon: float = 1.,
               sigma: float = 1.,
               min_distance: float = 0.,
               lambda_lj: float = 1.,
               linearize_below: Optional[float] = None,
               shift_energy: bool = False):
    """Initializer.

    Args:
      cutoff: above this value, the pairwise potential is set equal to 0. A
        cutoff is typically employed in simulation for performance reasons, so
        we also support this option here.
      box_length: array of shape [dim], side lengths of the simulation box. If
        None, the box length must be passed as an argument to the class methods.
      epsilon: determines the scale of the potential. See class docstring.
      sigma: determines the scale of the pairwise distance. See class docstring.
      min_distance: the pairwise distance is clipped to this value whenever it
        falls below it, which means that the pairwise potential is constant
        below this value. This is used for numerical reasons, to avoid the
        singularity at zero distance.
      lambda_lj: parameter for soft-core Lennard-Jones. Interpolates between
        a constant pairwise potential (lambda_lj = 0) and proper Lennard-Jones
        potential (lambda_lj = 1). See class docstring.
      linearize_below: below this value, the potential is linearized, i.e. it
        becomes a linear function of the pairwise distance. This can be used for
        numerical reasons, to avoid the potential growing too fast for small
        distances. If None, no linearization is done. NOTE: linearization
        removes the singularity at zero distance, but the pairwise force at
        zero distance is still undefined (becomes NaN). To have a well-defined
        force at zero distance, you need to set `min_distance > 0`.
      shift_energy: whether to shift the energy by a constant such that the
        potential is zero at the cutoff (spherically truncated and shifted LJ).
    """
    super().__init__(box_length, min_distance, linearize_below)
    self._cutoff = cutoff
    self._epsilon = epsilon
    self._sigma = sigma
    self._lambda_lj = lambda_lj
    self._shift = self._soft_core_lj_potential(cutoff**2) if shift_energy else 0

  def _soft_core_lj_potential(self, r2: Array) -> Array:
    r6 = r2**3 / self._sigma**6
    r6 += 0.5 * (1. - self._lambda_lj)**2
    r6inv = 1. / r6
    energy = r6inv * (r6inv - 1.)
    energy *= 4. * self._lambda_lj * self._epsilon
    return energy

  def _unclipped_pairwise_potential(self, r2: Array) -> Array:
    energy = self._soft_core_lj_potential(r2)
    # Apply radial cutoff and shift.
    energy = jnp.where(r2 <= self._cutoff**2, energy - self._shift, 0.)
    return energy
