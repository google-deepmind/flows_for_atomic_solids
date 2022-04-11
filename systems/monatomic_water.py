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

"""Monatomic (mW) water system."""

import math
from typing import Optional

import chex
from flows_for_atomic_solids.systems import energies
from flows_for_atomic_solids.utils import observable_utils as obs_utils
import jax
import jax.numpy as jnp

Array = chex.Array

MW_A = 7.049556277
MW_B = 0.6022245584
MW_GAMMA = 1.2
MW_EPSILON = 6.189  # Kcal/mol
MW_SIGMA = 2.3925  # Angstrom
MW_REDUCED_CUTOFF = 1.8
MW_COS = math.cos(109.47 / 180. * math.pi)
MW_LAMBDA = 23.15


class _TwoBodyEnergy(energies.PairwisePotentialEnergy):
  """Implements the two-body component of the monatomic-water energy."""

  def _unclipped_pairwise_potential(self, r2: Array) -> Array:
    r2 /= MW_SIGMA**2
    r = jnp.sqrt(r2)
    mask = jnp.array(r < MW_REDUCED_CUTOFF)
    # Distances on or above the cutoff can cause NaNs in the gradient of
    # `term_2` below, even though they're masked out in the forward computation.
    # To avoid this, we set these distances to a safe value.
    r = jnp.where(mask, r, 2. * MW_REDUCED_CUTOFF)
    term_1 = MW_A * MW_EPSILON * (MW_B / r2**2 - 1.)
    term_2 = jnp.where(mask, jnp.exp(1. / (r - MW_REDUCED_CUTOFF)), 0.)
    energy = term_1 * term_2
    return energy


class MonatomicWaterEnergy(energies.PotentialEnergy):
  """Evaluates the monatomic water energy with periodic boundary conditions.

  The monatomic water model, or mW model, consists of point particles that
  interact with each other via two-body interactions (between pairs of
  particles) and three-body interactions (between triplets of particles).

  The energy is decomposed as follows:
  ```
      energy = sum of all two-body interactions over distinct pairs +
               sum of all three-body interactions over distinct triplets
  ```
  More details on the specific functional form of the individual interaction
  terms can be found in the paper of Molinero and Moore (2009):
  https://arxiv.org/abs/0809.2811.
  """

  def __init__(self,
               box_length: Optional[Array] = None,
               min_distance: float = 0.,
               linearize_below: Optional[float] = None):
    """Constructor.

    Args:
      box_length: array of shape [dim], side lengths of the simulation box. If
        None, the box length must be passed as an argument to the class methods.
      min_distance: we clip the pairwise distance to this value in the
        calculation of the two-body term. This can be used to remove the
        singularity of the two-body term at zero distance.
      linearize_below: we linearize the two-body term below this value. If None,
        no linearization is done.
    """
    super().__init__(box_length)
    self._two_body_energy = _TwoBodyEnergy(
        min_distance=min_distance, linearize_below=linearize_below)

  def _three_body_energy(self, dr: Array) -> Array:
    """Compute three-body term for one sample.

    Args:
      dr: [num_particles, num_particles, 3] array of distance vectors
        between the particles.
    Returns:
      The three-body energy contribution of the sample (a scalar).
    """
    def _one_particle_contribution(dri: Array) -> Array:
      # dri is (num_particles-1) x 3.
      raw_norms = jnp.linalg.norm(dri, axis=-1)
      keep = raw_norms < MW_REDUCED_CUTOFF
      norms = jnp.where(keep, raw_norms, 1e20)
      norm_energy = jnp.exp(MW_GAMMA/(norms - MW_REDUCED_CUTOFF))
      norm_energy = jnp.where(keep, norm_energy, 0.)
      normprods = norms[None, :] * norms[:, None]
      # Note: the sum below is equivalent to:
      # dotprods = jnp.dot(dri, dri[..., None]).squeeze(-1)
      # but using jnp.dot results in loss of precision on TPU,
      # as evaluated by comparing to MD samples.
      dotprods = jnp.sum(dri[:, None, :] * dri[None, :, :], axis=-1)

      cos_ijk = dotprods / normprods

      energy = MW_LAMBDA * MW_EPSILON * (MW_COS - cos_ijk)**2
      energy *= norm_energy
      energy = jnp.triu(energy, 1)
      energy = jnp.sum(energy, axis=-1)
      return jnp.dot(energy, norm_energy)

    # Remove diagonal elements [i, i, :], changing the shape from
    # [num_particles, num_particles, 3] to [num_particles, num_particles-1, 3].
    clean_dr = jnp.rollaxis(jnp.triu(jnp.rollaxis(dr, -1), 1)[..., 1:]+
                            jnp.tril(jnp.rollaxis(dr, -1), -1)[..., :-1],
                            0, dr.ndim)
    # Vectorize over particles.
    energy = jnp.sum(jax.vmap(_one_particle_contribution)(clean_dr))
    return energy

  def energy(self,
             coordinates: Array,
             box_length: Optional[Array] = None) -> Array:
    """Computes energies for an entire batch of particles.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim], side lengths of the simulation
        box. If None, the default box length will be used instead.

    Returns:
      energy: array with shape [...] containing the computed energies.
    """
    if box_length is None:
      box_length = self.box_length
    dr = obs_utils.pairwise_difference_pbc(coordinates, box_length)
    dr /= MW_SIGMA
    two_body_energy = self._two_body_energy(coordinates, box_length)
    # Vectorize over samples.
    three_body_energy = jnp.vectorize(self._three_body_energy,
                                      signature='(m,m,n)->()')(dr)
    return two_body_energy + three_body_energy
