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

"""Constructors for particle models."""

from typing import Any, Mapping
import chex
import distrax
from flows_for_atomic_solids.models import bijectors
from flows_for_atomic_solids.models import particle_models
import numpy as np

Array = chex.Array


def make_particle_model(num_particles: int,
                        lower: Array,
                        upper: Array,
                        bijector: Mapping[str, Any],
                        base: Mapping[str, Any],
                        translation_invariant: bool,
                        ) -> particle_models.ParticleModel:
  """Constructs a particle model, with various configuration options.

  With N particles, the model is implemented as follows:
  1. We draw N particles randomly from a base distribution.
  2. We jointly transform the particles with a flow (a Distrax bijector).

  Optionally, the model can be made invariant to translations. We do this as
  follows:
  1. We draw N-1 particles and transform them with the flow as above.
  2. We add an extra particle at a fixed location.
  3. We choose a translation uniformly at random and apply it to all particles.

  Args:
    num_particles: number of particles.
    lower: array of shape [dim], the lower ranges of the box.
    upper: array of shape [dim], the upper ranges of the box.
    bijector: configures the bijector that transforms particles. Expected to
      have the following keys:
      * 'constructor': a callable that creates the bijector.
      * 'kwargs': keyword arguments to pass to the constructor.
    base: configures the base distribution. Expected to have the following keys:
      * 'constructor': a callable that creates the base distribution.
      * 'kwargs': keyword arguments to pass to the constructor.
    translation_invariant: if True, the model is constructed to be invariant
      to translations.

  Returns:
    A particle model.
  """
  if translation_invariant:
    num_mapped_particles = num_particles - 1
  else:
    num_mapped_particles = num_particles
  base_model = base['constructor'](
      num_particles=num_mapped_particles,
      lower=lower,
      upper=upper,
      **base['kwargs'])
  if len(np.unique(lower)) == 1 and len(np.unique(upper)) == 1:
    # Box is cubic.
    bij = bijector['constructor'](
        event_shape=base_model.event_shape,
        lower=lower[0],
        upper=upper[0],
        **bijector['kwargs'])
  else:
    # Box is rectangular.
    scaling_bijector = distrax.Block(
        bijectors.Rescale(lower_in=lower, upper_in=upper,
                          lower_out=0., upper_out=1.), 2)
    descaling_bijector = distrax.Inverse(scaling_bijector)
    bij = distrax.Chain([
        descaling_bijector,
        bijector['constructor'](
            event_shape=base_model.event_shape,
            lower=0.0,
            upper=1.0,
            **bijector['kwargs']),
        scaling_bijector,
    ])

  particle_model = particle_models.FlowModel(base_model, bij)
  if translation_invariant:
    particle_model = particle_models.TranslationInvariant(particle_model)
  return particle_model
