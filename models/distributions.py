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

"""Distrax distributions."""

from typing import Callable, Optional, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp

Array = chex.Array
Numeric = Union[Array, float]
PRNGKey = Array


class RejectionSampling(distrax.Distribution):
  """A rejection-sampling based distribution.

  Samples are drawn from the `proposal` distribution provided and are accepted
  only if they are valid as judged by `are_valid_fn`.

  NOTE: By default, the density of this distribution is unnormalized and doesn't
  integrate to unity. Specifically, calling `prob` returns the density of the
  proposal distribution if the sample is valid, otherwise it returns 0. This
  makes the normalizing constant equal to the acceptance rate. Optionally, the
  user can specify the normalizing constant, which will be used when computing
  probabilities.
  """

  def __init__(self,
               proposal: distrax.Distribution,
               are_valid_fn: Callable[[Array], Array],
               log_z: Optional[Numeric] = None):
    """Constructor.

    Args:
      proposal: the proposal distribution to sample from.
      are_valid_fn: checks whether samples are valid. If the input to the
        function is of shape `batch_shape + event_shape`, then the shape of the
        output should `batch_shape`.
      log_z: if specified, this value will be used as the log normalizing
        constant when computing probabilities. Should be equal to the log of
        the acceptance probability. Must be broadcastable to the distribution's
        batch shape.
    """
    super().__init__()
    self._proposal = proposal
    self._are_valid_fn = are_valid_fn
    self._log_z = 0. if log_z is None else log_z
    chex.assert_is_broadcastable(jnp.shape(self._log_z), self.batch_shape)

  @property
  def event_shape(self) -> Tuple[int, ...]:
    """Shape of event of distribution samples."""
    return self._proposal.event_shape

  @property
  def batch_shape(self) -> Tuple[int, ...]:
    """Batch shape of distribution samples."""
    return self._proposal.batch_shape

  def _sample_n_and_log_prob(self, key: PRNGKey, n: int) -> Tuple[Array, Array]:
    """Returns samples and log_probs obtained via rejection sampling."""
    # This function uses a jittable implementation of rejection sampling that
    # yields exact independent samples after a (random) number of iterations.

    def body(args):
      # In each iteration we propose `n` new samples and accept those that are
      # valid.
      key, samples, log_probs, accepted = args
      key, subkey = jax.random.split(key)
      proposals, proposal_log_probs = self._proposal.sample_and_log_prob(
          seed=subkey, sample_shape=n)
      valid_sample = self._are_valid_fn(proposals)
      valid_sample_event = valid_sample.reshape(valid_sample.shape + (1,) *
                                                len(self.event_shape))
      samples = jnp.where(valid_sample_event, proposals, samples)
      log_probs = jnp.where(valid_sample, proposal_log_probs, log_probs)
      accepted = valid_sample | accepted
      return key, samples, log_probs, accepted

    def cond(args):
      # If not all samples have been accepted yet, we continue iterating.
      _, _, _, accepted = args
      return ~jnp.all(accepted)

    samples = jnp.empty((n,) + self.batch_shape + self.event_shape)
    log_probs = jnp.empty((n,) + self.batch_shape)
    accepted = jnp.full((n,) + self.batch_shape, False)
    _, samples, log_probs, _ = jax.lax.while_loop(
        cond, body, (key, samples, log_probs, accepted))
    return samples, log_probs - self._log_z

  def _sample_n(self, key: PRNGKey, n: int) -> Array:
    samples, _ = self._sample_n_and_log_prob(key, n)
    return samples

  def log_prob(self, x: Array) -> Array:
    valid = self._are_valid_fn(x)
    log_prob = self._proposal.log_prob(x) - self._log_z
    return jnp.where(valid, log_prob, -jnp.inf)


class SphericalTruncation(RejectionSampling):
  """A rejection-sampling based distribution for spherical truncation.

  A sample from the `proposal` is accepted if its norm, computed across the
  entire event, is within the `cutoff`.
  """

  def __init__(self,
               proposal: distrax.Distribution,
               cutoff: float,
               log_z: Optional[Numeric] = None):
    """Constructor.

    Args:
      proposal: the proposal distribution to sample from.
      cutoff: the radial cutoff outside which samples are rejected.
      log_z: the log normalizing constant, same as in the parent class.
    """
    def are_within_cutoff(x: Array) -> Array:
      event_axes = list(range(-len(proposal.event_shape), 0))
      sq_dists = jnp.sum(x**2, axis=event_axes)
      return sq_dists <= cutoff**2

    super().__init__(proposal, are_within_cutoff, log_z)
