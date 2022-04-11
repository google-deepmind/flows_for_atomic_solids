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

"""Config file for a Lennard-Jones system in the solid FCC phase."""

from flows_for_atomic_solids.experiments import utils
from flows_for_atomic_solids.models import attention
from flows_for_atomic_solids.models import coupling_flows
from flows_for_atomic_solids.models import particle_model_constructors
from flows_for_atomic_solids.models import particle_models
from flows_for_atomic_solids.systems.lennard_jones import LennardJonesEnergy
import jax.numpy as jnp
from ml_collections import config_dict


FREQUENCIES = {
    32: 8,
    256: 16,
    500: 32,
}


def get_config(num_particles: int) -> config_dict.ConfigDict:
  """Returns the config."""
  box_lengths = utils.get_cubic_box_lengths(num_particles,
                                            density=1.28,
                                            dim=3)
  num_frequencies = FREQUENCIES[num_particles]

  config = config_dict.ConfigDict()
  config.state = dict(
      num_particles=num_particles,
      beta=0.5,
      lower=-box_lengths/2.,
      upper=box_lengths/2.,
  )
  conditioner = dict(
      constructor=coupling_flows.make_equivariant_conditioner,
      kwargs=dict(
          embedding_size=256,
          num_frequencies=num_frequencies,
          conditioner_constructor=attention.Transformer,
          conditioner_kwargs=dict(
              num_heads=2,
              num_layers=2,
              dropout_rate=0.,
              use_layernorm=False,
              w_init_final=jnp.zeros))
  )
  translation_invariant = True
  config.model = dict(
      constructor=particle_model_constructors.make_particle_model,
      kwargs=dict(
          bijector=dict(
              constructor=coupling_flows.make_split_coupling_flow,
              kwargs=dict(
                  num_layers=24,
                  num_bins=16,
                  conditioner=conditioner,
                  permute_variables=True,
                  split_axis=-1,
                  use_circular_shift=True,
                  prng=42,
              ),
          ),
          base=dict(
              constructor=particle_models.FaceCentredCubicLattice,
              kwargs=dict(
                  noise_scale=0.01,
                  remove_corner=translation_invariant,
              ),
          ),
          translation_invariant=translation_invariant,
      ),
  )
  shared_kwargs = dict(
      cutoff=2.7,
      box_length=box_lengths,
      lambda_lj=1.,
      shift_energy=True)
  config.train_energy = dict(
      constructor=LennardJonesEnergy,
      kwargs=dict(min_distance=0.01, linearize_below=0.8, **shared_kwargs)
  )
  config.test_energy = dict(
      constructor=LennardJonesEnergy,
      kwargs=dict(min_distance=0., **shared_kwargs)
  )
  config.train = dict(
      batch_size=128,
      learning_rate=7e-5,
      learning_rate_decay_steps=[250000, 500000],
      learning_rate_decay_factor=0.1,
      seed=42,
      max_gradient_norm=10000.,
  )
  config.test = dict(
      test_every=500,
      batch_size=2048,
  )
  return config
