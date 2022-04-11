#!/bin/bash
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

# This script installs the dependencies needed to run the training script for
# a flow modelling atomic solids.

python3 -m venv flows_for_atomic_solids_env
source flows_for_atomic_solids_env/bin/activate
pip install --upgrade pip
pip install -r flows_for_atomic_solids/requirements.txt

python -m flows_for_atomic_solids.experiments.train --system='lj_32' --num_iterations=0
