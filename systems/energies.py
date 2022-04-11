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

"""General-purpose energy functions."""

import abc
from typing import Optional

import chex
from flows_for_atomic_solids.utils import observable_utils as obs_utils
import jax
import jax.numpy as jnp

Array = chex.Array


def _check_same_dim(coordinates: Array, box_length: Array) -> None:
  """Check that `coordinates` and `box_length` have the same dimensionality."""
  if coordinates.ndim < 1:
    raise ValueError('The coordinates cannot be a scalar.')
  if box_length.ndim < 1:
    raise ValueError('The box length cannot be a scalar.')
  dim1 = coordinates.shape[-1]
  dim2 = box_length.shape[-1]
  if dim1 != dim2:
    raise ValueError(
        f'The dimensionality of the coordinates (got {dim1}) must be equal '
        f'to the dimensionality of the box (got {dim2}).')


class PotentialEnergy(abc.ABC):
  """Potential energy function of N interacting particles confined in a box.

  It assumes periodic boundary conditions.
  """

  def __init__(self, box_length: Optional[Array] = None):
    """Initializer.

    Args:
      box_length: array of shape [dim], the side lengths of the box that
        contains the particles. If None, the box length must be passed as an
        argument to the class methods.
    """
    if box_length is not None and box_length.ndim != 1:
      raise ValueError(
          f'`box_length` must be a vector. Got `box_length = {box_length}`.')
    self._box_length = box_length

  @property
  def box_length(self) -> Array:
    if self._box_length is None:
      raise ValueError('This class does not have a default box length.')
    return self._box_length

  def __call__(self,
               coordinates: Array,
               box_length: Optional[Array] = None) -> Array:
    return self.energy(coordinates, box_length)

  @abc.abstractmethod
  def energy(self,
             coordinates: Array,
             box_length: Optional[Array] = None) -> Array:
    """Computes the potential energy.

    Args:
      coordinates: array with shape [..., num_atoms, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim] containing the box length. If
        None, the default box length will be used instead.

    Returns:
      an array with shape [...] containing the computed energy values.
    """

  def forces(self,
             coordinates: Array,
             box_length: Optional[Array] = None) -> Array:
    """Computes the forces exerted on each particle for a batch of coordinates.

    Args:
      coordinates: array with shape [..., num_atoms, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim] containing the box length. If
        None, the default box length will be used instead.

    Returns:
      an array with shape [..., num_atoms, dim] whose [..., i, :] entry contains
      the total force exerted on particle i.
    """
    if box_length is None:
      box_length = self.box_length
    _check_same_dim(coordinates, box_length)
    grad_fn = jnp.vectorize(
        jax.grad(self.energy, argnums=0), signature='(n,d),(d)->(n,d)')
    return -grad_fn(coordinates, box_length)

  def pressure(self,
               coordinates: Array,
               box_length: Optional[Array] = None) -> Array:
    """Computes the excess virial pressure for a batch of coordinates.

    The virial pressure is computed according to the formula

       p_virial = (sum_i f_i . r_i - dim * V ∂U/∂V) / (dim * V)

    where r_i is the position of particle i, f_i is the total force acting on
    particle i, . is the dot-product, V the box volume and dim the number of
    dimensions. We assume that volume change occurs via a homogeneous affine
    expansion or contraction, so that

      dim * V ∂U/∂V = prod_i Li * ∂U/∂(Li)

    where Li is the i-th sidelength of the box. More details can be found in
    Thompson et al. (2009) https://doi.org/10.1063/1.3245303 (eqs. 1 and 13).

    Note that the above formula only contains the "excess" contribution due to
    the potential energy but not the "ideal" contribution due to the kinetic
    energy. The latter can be computed analytically.

    Args:
      coordinates: array with shape [..., num_atoms, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim] containing the box length. If
        None, the default box length will be used instead.

    Returns:
      an array with shape [...] containing the computed pressure.
    """
    if box_length is None:
      box_length = self.box_length
    _check_same_dim(coordinates, box_length)
    dim = coordinates.shape[-1]
    energy_grad = jnp.vectorize(
        jax.grad(self.energy, argnums=1), signature='(n,d),(d)->(d)')
    forces = self.forces(coordinates, box_length)
    virial = jnp.sum(coordinates * forces, axis=[-1, -2])
    virial -= jnp.sum(
        box_length * energy_grad(coordinates, box_length), axis=-1)
    return virial / (dim * jnp.prod(box_length, axis=-1))


class PairwisePotentialEnergy(PotentialEnergy):
  """Energy function based on a pairwise potential between particles.

  This is a base class for any energy function that can be written as a sum
  of pairwise potentials:

  E(x_1,...,x_N) = sum_{i<j} U(|x_i - x_j|^2).

  Distances are computed using periodic boundary conditions. The pairwise
  potential U should be implemented in a subclass.
  """

  def __init__(self,
               box_length: Optional[Array] = None,
               min_distance: float = 0.,
               linearize_below: Optional[float] = None):
    """Constructor.

    Args:
      box_length: array of shape [dim], the side lengths of the box that
        contains the particles. If None, the box length must be passed as an
        argument to the class methods.
      min_distance: the pairwise distance is clipped to this value whenever it
        falls below it, which means that the pairwise potential is constant
        below this value. This can be used for numerical reasons, for example to
        avoid singularities for potentials that diverge at zero distance.
      linearize_below: below this value, the potential is linearized, i.e. it
        becomes a linear function of the pairwise distance. This can be used for
        numerical reasons, for example when the potential grows too fast for
        small distances. If None, no linearization is done. NOTE: with
        linearization, the pairwise force at zero distance is undefined (becomes
        NaN). To have a well-defined force at zero distance, you need to set
        `min_distance > 0`.
    """
    super().__init__(box_length)
    self._min_distance = min_distance
    self._linearize_below = linearize_below

  @abc.abstractmethod
  def _unclipped_pairwise_potential(self, r2: Array) -> Array:
    """Scalar pairwise potential, to be implemented in a subclass.

    Args:
      r2: a scalar (0-dim array), the squared distance between particles.

    Returns:
      a scalar, the pairwise potential at that squared distance.
    """

  def _pairwise_potential(self, r2: Array) -> Array:
    """Numerically stable (clipped and linearized) version of the potential.

    Args:
      r2: a scalar (0-dim array), the squared distance between particles.

    Returns:
      a scalar, the pairwise potential at that squared distance.
    """
    r2 = jnp.clip(r2, self._min_distance ** 2)
    if self._linearize_below is None:
      return self._unclipped_pairwise_potential(r2)
    else:
      e0, g0 = jax.value_and_grad(
          lambda r: self._unclipped_pairwise_potential(r**2))(
              self._linearize_below)
      return jnp.where(
          r2 < self._linearize_below**2,
          e0 + (jnp.sqrt(r2) - self._linearize_below) * g0,
          self._unclipped_pairwise_potential(r2))

  def energy(self,
             coordinates: Array,
             box_length: Optional[Array] = None) -> Array:
    """Computes the potential energy.

    Args:
      coordinates: array with shape [..., num_atoms, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim] containing the box length. If
        None, the default box length will be used instead.

    Returns:
      an array with shape [...] containing the computed energy values.
    """
    if box_length is None:
      box_length = self.box_length
    _check_same_dim(coordinates, box_length)
    r2 = obs_utils.pairwise_squared_distance_pbc(coordinates, box_length)
    # We set the diagonal to a non-zero value to avoid infinities for pairwise
    # potentials that diverge at zero distances.
    r2 += jnp.eye(r2.shape[-1])
    energies = jnp.vectorize(self._pairwise_potential, signature='()->()')(r2)
    return jnp.sum(jnp.triu(energies, k=1), axis=[-2, -1])

  def pairwise_forces(self,
                      coordinates: Array,
                      box_length: Optional[Array] = None) -> Array:
    """Computes the pairwise forces for a batch of coordinates.

    Args:
      coordinates: array with shape [..., num_atoms, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim] containing the box length. If
        None, the default box length will be used instead.

    Returns:
      an array with shape [..., num_atoms, num_atoms, dim] whose [..., i, j, :]
      entry contains the force exerted on particle i from particle j.
    """
    if box_length is None:
      box_length = self.box_length
    _check_same_dim(coordinates, box_length)
    coordinate_deltas = obs_utils.pairwise_difference_pbc(
        coordinates, box_length)
    r2 = jnp.sum(coordinate_deltas**2, axis=-1)
    r2 += jnp.eye(r2.shape[-1])
    potential_gradient = jnp.vectorize(
        jax.grad(self._pairwise_potential), signature='()->()')
    forces = -2. * coordinate_deltas * potential_gradient(r2)[..., None]
    return forces

  def pairwise_pressure(self,
                        coordinates: Array,
                        box_length: Optional[Array] = None) -> Array:
    """Computes the excess virial pressure for a batch of coordinates.

    This function returns exactly the same result as `pressure` (up to numerical
    differences), but it's implemented in a way that is only valid for pairwise
    potentials. It's computed according to the formula

       p_virial = 1 / (dim * V) sum_i<j f_ij . r_ij,

    where r_ij is the distance vector between two particles, f_ij the force
    acting between this pair, . is the dot-product, V the box volume and dim the
    number of dimensions. More details can be found in Thompson et al. (2009)
    https://doi.org/10.1063/1.3245303 (see eqs. 1 and 7).

    Note that the above formula only contains the "excess" contribution due to
    the potential energy but not the "ideal" contribution due to the kinetic
    energy. The latter can be computed analytically.

    Args:
      coordinates: array with shape [..., num_atoms, dim] containing the
        particle coordinates.
      box_length: array with shape [..., dim] containing the box length. If
        None, the default box length will be used instead.

    Returns:
      an array with shape [...] containing the computed pressure.
    """
    if box_length is None:
      box_length = self.box_length
    _check_same_dim(coordinates, box_length)
    dim = coordinates.shape[-1]
    coordinate_deltas = obs_utils.pairwise_difference_pbc(
        coordinates, box_length)
    forces = self.pairwise_forces(coordinates, box_length)
    dot_product = jnp.sum(coordinate_deltas * forces, axis=-1)
    virial = jnp.sum(jnp.triu(dot_product, k=1), axis=[-2, -1])
    pressure = virial / (dim * jnp.prod(box_length, axis=-1))
    return pressure
