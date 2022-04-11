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

"""Particle models."""

import abc
import math
from typing import Optional, Tuple

import chex
import distrax
from flows_for_atomic_solids.models import distributions
from flows_for_atomic_solids.utils import lattice_utils
from flows_for_atomic_solids.utils import observable_utils as obs_utils
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

Array = chex.Array
PRNGKey = Array


class ParticleModel(distrax.Distribution, metaclass=abc.ABCMeta):
  """A distribution over particles confined in a box.

  It assumes that the box has periodic boundary conditions (it's a torus).

  A sample from this distribution is a set of N particles in a D-dimensional
  box. Each particle is described by a vector of D position coordinates. A
  sample from this distribution has shape [N, D].
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array):
    """Constructor.

    Args:
      num_particles: number of particles.
      lower: array of shape [dim], the lower ranges of the box.
      upper: array of shape [dim], the upper ranges of the box.
    """
    if num_particles < 1:
      raise ValueError(
          f'The number of particles must be at least 1, got {num_particles}.')
    if lower.ndim != 1:
      raise ValueError(f'`lower` must have one array dimension, '
                       f'got `lower.ndim = {lower.ndim}`.')
    if upper.ndim != 1:
      raise ValueError(f'`upper` must have one array dimension, '
                       f'got `upper.ndim = {upper.ndim}`.')
    (dim,) = lower.shape
    if upper.shape != (dim,):
      raise ValueError(
          f'`lower` and `upper` must have the same shape. Got '
          f'`lower.shape = {lower.shape}` and `upper.shape = {upper.shape}`.')
    if np.any(lower >= upper):
      raise ValueError(
          f'`lower` components must be less than `upper` components. '
          f'Got `lower == {lower}` and `upper == {upper}`.')
    self._num_particles = num_particles
    self._dim = dim
    self._lower = lower
    self._upper = upper
    self._width = upper - lower
    super().__init__()

  @property
  def num_particles(self) -> int:
    return self._num_particles

  @property
  def dim(self) -> int:
    return self._dim

  @property
  def lower(self) -> Array:
    return self._lower

  @property
  def upper(self) -> Array:
    return self._upper

  @property
  def width(self) -> Array:
    return self._width

  @property
  def event_shape(self) -> Tuple[int, int]:
    return (self._num_particles, self._dim)

  def wrap(self, x: Array) -> Array:
    """Wraps `x` back into the box."""
    return jnp.mod(x - self._lower, self._width) + self._lower

  def log_prob(self, particles: Array) -> Array:
    if particles.shape[-2:] != self.event_shape:
      raise ValueError(
          f'Events of shape {particles.shape[-2:]} were passed to `log_prob`,'
          f' but `{self.name}` expects events of shape {self.event_shape}.')
    return self._log_prob_no_checks(particles)

  def _log_prob_no_checks(self, particles: Array) -> Array:
    """Called by `log_prob`. Should be implemented in a subclass."""
    raise NotImplementedError('`log_prob` is not implemented for '
                              f'`{self.name}`.')


class FlowModel(ParticleModel):
  """A particle model transformed by a flow.

  It takes in a base model and a flow (a Distrax bijector). The model is the
  pushforward of the base model through the flow.

  If the base model is invariant to particle permutations and the flow is
  equivariant to particle permutations, the resulting model will be invariant to
  particle permutations too.
  """

  def __init__(self, base_model: ParticleModel, bijector: distrax.BijectorLike):
    """Constructor.

    Args:
      base_model: the base model to transform with the flow.
      bijector: the flow, a Distrax bijector.
    """
    super().__init__(base_model.num_particles, base_model.lower,
                     base_model.upper)
    self._flow_model = distrax.Transformed(base_model, bijector)

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    return self._flow_model.sample(seed=key, sample_shape=n)

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    return self._flow_model.sample_and_log_prob(seed=key, sample_shape=n)

  def _log_prob_no_checks(self, particles: Array) -> Array:
    return self._flow_model.log_prob(particles)


class TranslationInvariant(ParticleModel):
  """Translation-invariant particle model: works by inserting an extra particle.

  This model takes a base model with N particles, and produces a translation-
  invariant model with N+1 particles. It works as follows:
  1. We draw N particles from the base model.
  2. We add an extra particle at a fixed location.
  3. We choose a translation uniformly at random and apply it to all particles.

  The probability density of N+1 particles `x` is `p(x) = p(z) p(u)`, where:
  - `u` is the random translation.
  - `p(u) = 1 / (upper - lower) ^ dim` is the uniform density on the box.
  - `z` are the N particles before translation by `u`.
  - `p(z)` is the density of the base model.

  NOTE: the above procedure breaks permutation invariance. If the base model is
  permutation-invariant, the resulting distribution over N+1 particles will be
  invariant to permutations of the first N particles, but not to permutations
  of all N+1 particles.
  """

  def __init__(self, base_model: ParticleModel):
    """Constructor.

    Args:
      base_model: The base model. The number of particles of the translation-
        invariant model is the number of particles of the base model plus one.
    """
    super().__init__(base_model.num_particles + 1, base_model.lower,
                     base_model.upper)
    # We place the extra particle at the corner of the box.
    self._extra_particle = self.lower
    # Log density of the random shift: log uniform on the box.
    self._shift_log_prob = -np.sum(np.log(self.width))
    self._base_model = base_model

  def _add_particle(self, key: PRNGKey, particles: Array) -> Array:
    batch_shape = particles.shape[:-2]
    extra = jnp.tile(self._extra_particle, batch_shape + (1, 1))
    particles = jnp.concatenate((particles, extra), axis=-2)
    shift = self.width * jax.random.uniform(key, batch_shape + (1, self.dim))
    particles = self.wrap(particles + shift)
    return particles

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    key1, key2 = jax.random.split(key)
    samples = self._base_model.sample(seed=key1, sample_shape=n)
    samples = self._add_particle(key2, samples)
    return samples

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    key1, key2 = jax.random.split(key)
    samples, log_prob = self._base_model.sample_and_log_prob(
        seed=key1, sample_shape=n)
    samples = self._add_particle(key2, samples)
    log_prob += self._shift_log_prob
    return samples, log_prob

  def _log_prob_no_checks(self, particles: Array) -> Array:
    shift = particles[..., -1:, :] - self._extra_particle
    particles = self.wrap(particles[..., :-1, :] - shift)
    return self._base_model.log_prob(particles) + self._shift_log_prob


def _log_factorial(n: int) -> float:
  return sum(math.log(i + 1) for i in range(n))


class Lattice(ParticleModel, metaclass=abc.ABCMeta):
  """A particle model based on a lattice for sampling particles.

  Samples are generated by adding random noise to the lattice followed by a
  random permutation of particles. The noise is drawn i.i.d. from a distribution
  that is defined on a suitable interval such that perturbed particles never
  escape their lattice sites.
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array,
               noise_scale: float,
               cell_aspect: Array,
               atom_positions_in_cell: Array,
               remove_corner: bool,
               spherical_noise: bool):
    """Constructor.

    Args:
      num_particles: number of particles.
      lower: array of shape [dim], the lower ranges of the box.
      upper: array of shape [dim], the upper ranges of the box.
      noise_scale: scale for the noise distribution.
      cell_aspect: vector of length `dim` with the relative length of each
        of the unit cell axes.
      atom_positions_in_cell: [N x dim] matrix of fractional coordinates of
        atoms within each unit cell (N being the number of atoms per cell).
      remove_corner: whether the lattice excludes one particle, at the
        coordinate origin. Notice that if True, `num_particles` should not
        count the missing particle, e.g., a 2x2x2 simple cubic lattice should
        have `num_particles` equal 7 and not 8 if `remove_corner` is True.
      spherical_noise: whether to cut off the noise spherically or
        indepedently across each axis.
    """
    super().__init__(num_particles, lower, upper)
    if noise_scale < 0.:
      raise ValueError(
          f'`noise_scale` can\'t be negative; got {noise_scale}.')
    self._cell_aspect = cell_aspect
    self._atom_positions_in_cell = atom_positions_in_cell
    self._remove_corner = remove_corner
    self._lattice, self._lattice_constant, self._repeats = self._make_lattice()
    self._spherical_noise = spherical_noise
    self._noise_dist = self._make_noise_dist(noise_scale, spherical_noise)
    self._log_num_permutations = _log_factorial(num_particles)

  @property
  def lattice(self) -> Array:
    return self._lattice

  @property
  def lattice_constant(self) -> Array:
    return self._lattice_constant

  def _make_lattice(self) -> Tuple[Array, Array, Array]:
    """Returns a lattice and its lattice constant."""
    lattice, lc, repeats = lattice_utils.make_lattice(
        self.lower, self.upper, self._cell_aspect,
        self._atom_positions_in_cell, self.num_particles + self._remove_corner)
    if self._remove_corner:
      # Make sure the removed site is at the lower corner
      lattice = lattice - lattice[0] + self.lower
      lattice = lattice[1:]
    lattice = self.wrap(lattice)
    return lattice, lc, repeats

  def _get_lattice_cutoff(self) -> float:
    r_matrix = obs_utils.pairwise_distance_pbc(self.lattice, self.width)
    r_matrix = r_matrix + jnp.eye(self.num_particles) * r_matrix.max()
    cutoff = r_matrix.min() / 2
    return cutoff

  def _make_noise_dist(
      self, noise_scale: float, spherical: bool) -> distrax.Distribution:
    """Returns a distribution to sample the noise from."""
    if spherical:
      # Find closest lattice points, cutoff is half of that distance
      cutoff = self._get_lattice_cutoff()
      proposal = distrax.Normal(
          loc=jnp.zeros((self.num_particles, self.dim)),
          scale=noise_scale)
      proposal = distrax.Independent(proposal, 1)
      # We rejection-sample for each particle independently.
      log_z = tfp.distributions.Chi(self.dim).log_cdf(cutoff / noise_scale)
      proposal = distributions.SphericalTruncation(proposal, cutoff, log_z)
      return distrax.Independent(proposal, 1)
    else:
      # Independent noise up to half the smallest distance along each axis
      dx = jnp.abs(obs_utils.pairwise_difference_pbc(self.lattice, self.width))
      # Ignore zeros when computing minimum distance (this is a heuristic
      # to obtain the maximal non-overlapping noise - we presume that
      # atoms that coincide on one axis are well separated on some other axis)
      dx = jnp.where(dx < 1e-9, np.amax(dx), dx)
      cutoffs = jnp.amin(dx.reshape(-1, dx.shape[-1]), axis=0) / 2.0
      # TruncatedNormal does not allow different truncation cutoffs for each
      # axis, so we scale down the std dev by the cutoffs and then scale up the
      # result by the same amount.
      proposal = tfp.distributions.TruncatedNormal(
          loc=jnp.zeros((self.num_particles, self.dim)),
          scale=noise_scale / cutoffs,
          low=-1.0,
          high=1.0)
      return distrax.Independent(
          distrax.Transformed(proposal, distrax.ScalarAffine(0., cutoffs)),
          2)

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    keys = jax.random.split(key, 1 + n)
    noise, log_prob = self._noise_dist.sample_and_log_prob(
        sample_shape=n, seed=keys[0])
    samples = self.wrap(noise + self._lattice)
    samples = jax.vmap(jax.random.permutation)(keys[1:], samples)
    # The probability density p(x) is the average of p(x|z), where z is a
    # particle permutation. But because we assume perturbed particles never
    # escape their lattice cell, all but one of p(x|z) are zero, so the average
    # reduces to a division by the number of permutations.
    log_prob -= self._log_num_permutations
    return samples, log_prob

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    samples, _ = self._sample_n_and_log_prob(key, n)
    return samples

  def _wrap_displacement(self, dx: Array) -> Array:
    """Returns the difference vector to the nearest image under PBCs."""
    return dx - self.width * jnp.round(dx / self.width)

  def _get_nearest_lattice_indices(self, x: Array) -> Array:
    """Returns indices of the nearest lattice sites."""
    deltas = self.lattice[..., :, None, :] - x[..., None, :, :]
    deltas = self._wrap_displacement(deltas)
    sq_dist = jnp.sum(deltas**2, axis=-1)
    return jnp.argmin(sq_dist, axis=-2)

  def _check_single_occupation(self, indices: Array) -> bool:
    """Returns True if each lattice index appears once otherwise False."""
    index_error = jnp.sort(indices) - jnp.arange(self.num_particles)
    return jnp.all(index_error == 0)

  def _log_prob_no_checks(self, particles: Array) -> Array:
    if not self._spherical_noise:
      raise NotImplementedError('The log_prob for non-spherical noise is not '
                                'yet implemented.')
    indices = self._get_nearest_lattice_indices(particles)
    is_valid = jnp.vectorize(self._check_single_occupation,
                             signature='(m)->()')(indices)
    noise = self._wrap_displacement(particles - self.lattice[indices])
    log_prob = self._noise_dist.log_prob(noise) - self._log_num_permutations
    return jnp.where(is_valid, log_prob, -jnp.inf)


class SimpleCubicLattice(Lattice):
  """A particle model based on a simple cubic lattice for sampling particles.

  The noise is drawn i.i.d. from a truncated Gaussian that is defined on a
  suitable interval such that perturbed particles never escape their lattice
  cell.
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array,
               noise_scale: float,
               remove_corner: bool = False,
               cell_aspect: Optional[Array] = None):
    dim = lower.shape[-1]
    if cell_aspect is None:
      cell_aspect = np.ones(dim)
    atom_positions_in_cell = np.ones((1, dim)) * 0.5
    spherical_noise = False
    super().__init__(num_particles, lower, upper, noise_scale, cell_aspect,
                     atom_positions_in_cell, remove_corner, spherical_noise)

  def _log_prob_no_checks(self, particles: Array) -> Array:
    if self._remove_corner:
      raise NotImplementedError(
          '`_log_prob_no_checks` not implemented in '
          f'{self.name} when `remove_corner` is True.')
    noise = jnp.mod(particles - self.lower, self._lattice_constant)
    noise -= self._lattice_constant / 2.
    log_prob = self._noise_dist.log_prob(noise)
    log_prob -= self._log_num_permutations
    bins = [l + jnp.arange(r + 1) * lc for (l, r, lc) in zip(
        self.lower, self._repeats.tolist(), self.lattice_constant)]

    def has_no_overlap(x):
      hist, _ = jnp.histogramdd(x, bins=bins)
      return jnp.all(hist == 1)

    no_overlap = jnp.vectorize(has_no_overlap, signature='(m,n)->()')(particles)
    in_range = (particles >= self.lower) & (particles <= self.upper)
    in_range = jnp.all(in_range, axis=[-2, -1])
    return jnp.where(in_range & no_overlap, log_prob, -jnp.inf)


class FaceCentredCubicLattice(Lattice):
  """A particle model based on a face centred cubic (FCC) lattice.

  We construct the lattice by translating a unit cell containing four atoms. We
  then draw i.i.d. noise for each particle from a spherically truncated normal.
  The support of this distribution is chosen such that the noise distributions
  of neighbouring lattice sites are pairwise disjoint.

  Note that this lattice requires three spatial dimensions (`dim == 3`).
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array,
               noise_scale: float,
               remove_corner: bool):
    """Constructor.

    Args:
      num_particles: number of particles.
      lower: array of shape [dim], the lower ranges of the box.
      upper: array of shape [dim], the upper ranges of the box.
      noise_scale: scale for the noise distribution.
      remove_corner: if True, we remove the lattice site located at the corner
        of the box.
    """
    dim = lower.shape[-1]
    if dim != 3:
      raise ValueError(f'Expected the box dimensionality to be 3, got {dim}.')
    cell_aspect = np.ones(dim)
    atom_positions_in_cell = np.array([[0., 0., 0.],
                                       [1., 1., 0.],
                                       [1., 0., 1.],
                                       [0., 1., 1.]]) / 2
    spherical_noise = True
    super().__init__(num_particles, lower, upper, noise_scale, cell_aspect,
                     atom_positions_in_cell, remove_corner, spherical_noise)


class DiamondCubicLattice(Lattice):
  """A particle model based on a diamond cubic lattice.

  The unit cell of a diamond cubic lattice contains eight atoms: the four atoms
  of the FCC unit cell and an additional four atoms that are located within the
  unit cell. See here for more details on the mathematical structure and an
  illustration: https://en.wikipedia.org/wiki/Diamond_cubic.

  Note that this lattice requires three spatial dimensions (`dim == 3`).
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array,
               noise_scale: float,
               remove_corner: bool):
    """Constructor.

    Args:
      num_particles: number of particles.
      lower: array of shape [dim], the lower ranges of the box.
      upper: array of shape [dim], the upper ranges of the box.
      noise_scale: scale for the noise distribution.
      remove_corner: if True, we remove the lattice site located at the corner
        of the box.
    """
    dim = lower.shape[-1]
    if dim != 3:
      raise ValueError(f'Expected the box dimensionality to be 3, got {dim}.')
    cell_aspect = np.ones(dim)
    atom_positions_in_cell = np.array([
        # The first 4 atoms are the same as for FCC.
        [0., 0., 0.],
        [0., 2., 2.],
        [2., 0., 2.],
        [2., 2., 0.],
        # The additional are located within the unit cell.
        [3., 3., 3.],
        [3., 1., 1.],
        [1., 3., 1.],
        [1., 1., 3.],
    ]) / 4
    spherical_noise = True
    super().__init__(num_particles, lower, upper, noise_scale, cell_aspect,
                     atom_positions_in_cell, remove_corner, spherical_noise)


class HexagonalIceLattice(Lattice):
  """A particle model based on a hexagonal ice (Ice Ih) lattice.

  Note that this lattice requires three spatial dimensions (`dim == 3`).
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array,
               noise_scale: float,
               remove_corner: bool):
    """Constructor.

    Args:
      num_particles: number of particles.
      lower: an array of shape [dim], the lower ranges of the box.
      upper: an array of shape [dim], the upper ranges of the box.
      noise_scale: scale for the noise distribution.
      remove_corner: if True, we remove the lattice site located at the corner
        of the box.
    """
    dim = lower.shape[-1]
    if dim != 3:
      raise ValueError(f'Expected the box dimensionality to be 3, got {dim}.')
    # According to http://people.cs.uchicago.edu/~ridg/digbio12/primerice.pdf
    # within the non-orthogonal base cell:
    #
    # base = np.array([[1, 0, 0],
    #                  [-0.5, np.sqrt(3)/2, 0],
    #                  [0, 0, 2*np.sqrt(6)/3]])
    #
    # we need 4 atoms at absolute coordinates:
    #
    # atom_coords = np.array([[0.5, np.sqrt(3)/6, np.sqrt(6)/24],
    #                         [1.0, np.sqrt(3)/3, 15*np.sqrt(6)/24],
    #                         [0.5, np.sqrt(3)/6, 7*np.sqrt(6)/24],
    #                         [1.0, np.sqrt(3)/3, 8*np.sqrt(6)/2]])
    #
    # We make an orthogonal cell from integer multiples of the
    # non-orthogonal base:
    #
    # ortho_base = np.array([[1, 0, 0],
    #                        [1, 2, 0],
    #                        [0, 0, 1]]) @ base
    #
    # This orthogonal cell has twice the volume of the non-orthogonal base
    # and the relative atom coordinates within it can be easily found by
    # replicating the non-orthogonal cell and retaining only atoms within
    # the orthogonal one. This results in the 8 atom positions below.
    cell_aspect = np.array([1.0, np.sqrt(3), np.sqrt(8/3)])
    a = 6 * 0.0625
    atom_positions_in_cell = np.array([
        [3., 5., 3 + a],
        [0., 4., 0 + a],
        [0., 2., 3 + a],
        [3., 1., 0 + a],
        [0., 2., 6 - a],
        [3., 1., 3 - a],
        [3., 5., 6 - a],
        [0., 4., 3 - a],
    ]) / 6.0
    spherical_noise = True
    super().__init__(num_particles, lower, upper, noise_scale, cell_aspect,
                     atom_positions_in_cell, remove_corner, spherical_noise)


class HalfDiamondLattice(Lattice):
  """A particle model based on a half-cell diamond lattice.

  The unit cell is 1/2 of a standard diamond cubic cell (and therefore has
  4 atoms per unit cell).
  If the base vectors of the diamond cubic cell are a=(1, 0, 0), b=(0, 1, 0)
  and c=(0, 0, 1), this cell has basis a'=(1/2, 1/2, 0), b'=(1/2, -1/2, 0),
  and c'=c.

  Note that this lattice requires three spatial dimensions (`dim == 3`).
  """

  def __init__(self,
               num_particles: int,
               lower: Array,
               upper: Array,
               noise_scale: float,
               remove_corner: bool):
    """Constructor.

    Args:
      num_particles: number of particles.
      lower: array of shape [dim], the lower ranges of the box.
      upper: array of shape [dim], the upper ranges of the box.
      noise_scale: scale for the noise distribution.
      remove_corner: if True, we remove the lattice site located at the corner
        of the box.
    """
    dim = lower.shape[-1]
    if dim != 3:
      raise ValueError(f'Expected the box dimensionality to be 3, got {dim}.')
    cell_aspect = np.array([1.0, 1.0, np.sqrt(2.0)])
    # Half a diamond cubic cell by using base (0.5, 0.5) and (-0.5, 0.5)
    # in the x-y plane instead of (1, 0) and (0, 1)
    atom_positions_in_cell = np.array([
        [0, 0, 0],
        [0, 2, 1],
        [2, 2, 2],
        [2, 0, 3]
    ]) / 4.0
    spherical_noise = True
    super().__init__(num_particles, lower, upper, noise_scale, cell_aspect,
                     atom_positions_in_cell, remove_corner, spherical_noise)
