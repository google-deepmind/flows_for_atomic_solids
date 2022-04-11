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

"""Utils for computing observables in atomistic systems."""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import scipy.special

Array = chex.Array


def _volume_sphere(radius: Array, dim: int) -> Array:
  """Computes the volume of a Euclidean ball in `dim` dimensions."""
  c = jnp.pi**(dim/2) / scipy.special.gamma(1 + dim/2)  # Volume of unit sphere.
  return c * radius**dim


def _check_same_dim(coordinates: Array, box_length: Array) -> None:
  """Check that `coordinates` and `box_length` have equal spatial dimension."""
  if coordinates.ndim < 1:
    raise ValueError('The coordinates cannot be a scalar.')
  if box_length.ndim < 1:
    raise ValueError('The box length cannot be a scalar.')
  dim_coords = coordinates.shape[-1]
  dim_box = box_length.shape[-1]
  if dim_coords != dim_box:
    raise ValueError(
        f'The dimensionality of the coordinates (got {dim_coords}) must be '
        f'equal to the dimensionality of the box (got {dim_box}).')


def _pairwise_difference(coordinates: Array) -> Array:
  """Computes pairwise difference vectors.

  Args:
    coordinates: array with shape [..., num_particles, dim] containing particle
      coordinates.

  Returns:
    Array with shape [..., num_particles, num_particles, dim], difference
    vectors for all pairs of particles.
  """
  if coordinates.ndim < 2:
    raise ValueError(
        f'Expected at least 2 array dimensions, got {coordinates.ndim}.')
  x = coordinates[..., :, None, :]
  y = coordinates[..., None, :, :]
  return x - y


def pairwise_difference_pbc(coordinates: Array, box_length: Array) -> Array:
  """Computes pairwise distance vectors obeying periodic boundary conditions.

  Args:
    coordinates: array with shape [..., num_particles, dim] containing particle
      coordinates.
    box_length: array with shape [..., dim], the edge lengths of the box.

  Returns:
    Array with shape [..., num_particles, num_particles, dim], the pairwise
    distance vectors with respect to periodic boundary conditions.
  """
  _check_same_dim(coordinates, box_length)
  deltas = _pairwise_difference(coordinates)
  chex.assert_is_broadcastable(box_length.shape[:-1], coordinates.shape[:-2])
  box_length = box_length[..., None, None, :]
  return deltas - box_length * jnp.round(deltas / box_length)


def squared_distance_pbc(coordinate_deltas: Array, box_length: Array) -> Array:
  """Computes squared distance obeying periodic boundary conditions.

  Args:
    coordinate_deltas: array with shape [..., dim] containing difference
      vectors.
    box_length: array with shape [..., dim], the edge lengths of the box.

  Returns:
    Array with shape [...], the squared distances with respect to periodic
    boundary conditions.
  """
  _check_same_dim(coordinate_deltas, box_length)
  chex.assert_is_broadcastable(
      box_length.shape[:-1], coordinate_deltas.shape[:-1])
  coordinate_deltas_pbc = (coordinate_deltas - box_length *
                           jnp.round(coordinate_deltas / box_length))
  return jnp.sum(coordinate_deltas_pbc**2, axis=-1)


def pairwise_squared_distance_pbc(coordinates: Array,
                                  box_length: Array) -> Array:
  """Computes pairwise squared distance obeying periodic boundary conditions.

  Args:
    coordinates: array with shape [..., num_particles, dim] containing particle
      coordinates.
    box_length: array with shape [..., dim], the edge lengths of the box.

  Returns:
    Array with shape [..., num_particles, num_particles] with pairwise squared
    distances.
  """
  _check_same_dim(coordinates, box_length)
  coordinate_deltas = _pairwise_difference(coordinates)
  chex.assert_is_broadcastable(box_length.shape[:-1], coordinates.shape[:-2])
  return squared_distance_pbc(coordinate_deltas, box_length[..., None, None, :])


def pairwise_distance_pbc(coordinates: Array, box_length: Array) -> Array:
  """Computes pairwise distance obeying periodic boundary conditions.

  Args:
    coordinates: array with shape [..., num_particles, dim] containing particle
      coordinates.
    box_length: array with shape [..., dim], the edge lengths of the box.

  Returns:
    Array of shape [..., num_particles, num_particles] with pairwise distances.
  """
  return jnp.sqrt(pairwise_squared_distance_pbc(coordinates, box_length))


def radial_distribution_function(coordinates: Array,
                                 box_length: Array,
                                 num_bins: int = 300) -> Array:
  """Computes the radial distribution function.

  The radial distribution function `g(r)`, also known as the pair correlation
  function, quantifies the variation in density as a function of distance with
  respect to a reference particle.

  Args:
    coordinates: array with shape [..., num_particles, dim] containing particle
      coordinates.
    box_length: array with shape [dim], the edge lengths of the simulation box.
    num_bins: the number of bins (resolution) for computing the radial
      distribution function.
  Returns:
    gr: array with shape [num_bins, 2] where the first column is the centre
      of the bin, `r`, and the second the estimated function value `g(r)`.
  """
  num_particles, dim = coordinates.shape[-2:]
  if jnp.shape(box_length) != (dim,):
    raise ValueError(
        f'`box_length` must be a vector of length {dim}.'
        f' Got `box_length = {box_length}`.')
  min_box_length = jnp.min(box_length)
  box_volume = jnp.product(box_length)

  coordinates = jnp.reshape(coordinates, [-1, num_particles, dim])
  batch_size = coordinates.shape[0]
  dr = pairwise_distance_pbc(coordinates, box_length)
  # Extract all upper triangular matrix elements.
  dr = jax.vmap(lambda x: x[jnp.triu_indices(x.shape[0], k=1)])(dr)
  hist, bins = jnp.histogram(dr, bins=num_bins, range=(0, min_box_length / 2.))

  density = num_particles / box_volume
  volume_shell = _volume_sphere(bins[1:], dim) - _volume_sphere(bins[:-1], dim)
  # Normaliser for histogram so that `g(r)` approaches unity for large `r`.
  normaliser = volume_shell * density * batch_size * (num_particles - 1) / 2
  gr = jnp.column_stack(((bins[:-1] + bins[1:]) / 2, hist / normaliser))
  return gr


def compute_histogram(data: Array,
                      num_bins: int = 100,
                      data_range: Optional[Tuple[float, float]] = None,
                      density: bool = True) -> Array:
  """Returns a histogram of the input data."""
  hist, bins = jnp.histogram(
      data, bins=num_bins, density=density, range=data_range)
  centers = (bins[:-1] + bins[1:]) / 2
  return jnp.column_stack((centers, hist))


def free_energy_fep(forward: Array,
                    beta: float) -> Array:
  """Returns the FEP estimate of the free energy difference.

  The free energy estimate is computed using the Free Energy Perturbation (FEP)
  estimator (for details see doi.org/10.1063/1.1740409). This is an importance
  sampling based estimator that requires specification of forward values.

  Args:
    forward: a one-dimensional array that contains the forward work values.
    beta: the inverse temperature.
  Returns:
    the estimated free energy difference.
  """
  log_sum_exp = jax.scipy.special.logsumexp(-beta * forward)
  log_average_exp = log_sum_exp - jnp.log(forward.size)
  return - log_average_exp / beta


def free_energy_fep_running(forward: Array,
                            beta: float,
                            num_evals: int = 10) -> Array:
  """Returns a running average of the FEP estimate.

  Args:
    forward: a one-dimensional array that contains the forward work values.
    beta: the inverse temperature.
    num_evals: number of times the free energy difference is evaluated.
  Returns:
    an array with shape [num_evals, 2] where the first column corresponds to the
    number of samples and the second column to the estimated free energy
    difference.
  """
  num_samples = forward.size
  sample_increment = num_samples // num_evals
  running_average = []
  for i in range(num_evals):
    samples_i = sample_increment * (i + 1)
    df_i = free_energy_fep(forward[:samples_i], beta)
    running_average.append((samples_i, df_i))
  return jnp.array(running_average)


def _compute_importance_weights(model_log_probs: Array,
                                target_log_probs: Array) -> Array:
  """Returns the normalised importance weights.

  Args:
    model_log_probs: an array containing the log_probs computed by the model.
    target_log_probs: an array containing the target log_probs.
  Returns:
    the normalised importance weights.
  """
  assert model_log_probs.shape == target_log_probs.shape
  # Make sure all computations are done in double precision.
  model_log_probs = model_log_probs.astype(jnp.float64)
  target_log_probs = target_log_probs.astype(jnp.float64)
  # Compute the self-normalised importance weights.
  return jax.nn.softmax(target_log_probs - model_log_probs, axis=None)


def compute_ess(model_log_probs: Array, target_log_probs: Array) -> Array:
  """Returns the standard estimate of the effective sample size (ESS).

  More details can be found in https://arxiv.org/pdf/1602.03572.pdf (see Eq. 6).

  Args:
    model_log_probs: an array containing the log_probs computed by the model.
    target_log_probs: an array containing the target log_probs.
  Returns:
    the ESS as a percentage between 0 and 100.
  """
  weights = _compute_importance_weights(model_log_probs, target_log_probs)
  return 100 / (jnp.sum(weights**2) * weights.size)


def compute_logz(model_log_probs: Array, target_log_probs: Array) -> Array:
  """Returns an estimate of the logarithm of the ratio of normalisers.

  Args:
    model_log_probs: an array containing the log_probs computed by the model.
    target_log_probs: an array containing the target log_probs.
  Returns:
    the estimated difference of the log normalisers.
  """
  assert model_log_probs.shape == target_log_probs.shape
  log_prob_diff = target_log_probs - model_log_probs
  log_sum_exp = jax.scipy.special.logsumexp(log_prob_diff, axis=None)
  return log_sum_exp - jnp.log(target_log_probs.size)


def compute_importance_estimate(x: Array,
                                model_log_probs: Array,
                                target_log_probs: Array) -> Array:
  """Return an importance sampling estimate of the mean value of `x`."""
  weights = _compute_importance_weights(model_log_probs, target_log_probs)
  if weights.shape != x.shape:
    raise ValueError('The shape of the importance sampling weights '
                     f'{weights.shape} differs from the shape of the data '
                     f'{x.shape} but is expected to be the same.')
  return jnp.sum(weights * x)
