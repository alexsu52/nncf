# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import Optional

import torch
import torch.fx
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme
from nncf.data import Dataset
from nncf.experimental.torch_fx.transformations import merge_conv_and_bias
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.scopes import IgnoredScope

DEFAULT_RANGE_TYPE = "mean_min_max"


def quantize_impl(
    model: torch.fx.GraphModule,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> torch.nn.Module:
    """
    Implementation of the `quantize()` method for the Torch FX backend.
    """
    if fast_bias_correction is False:
        raise ValueError(f"fast_bias_correction={fast_bias_correction} is not supported")
    if target_device == TargetDevice.CPU_SPR:
        raise nncf.InternalError("target_device == CPU_SPR is not supported")
    if mode is not None:
        raise ValueError(f"mode={mode} is not supported")

    original_graph_meta = model.meta

    copied_model = deepcopy(model)

    if advanced_parameters is None:
        advanced_parameters = AdvancedQuantizationParameters()
    # torch.fx supports only assymetric activations quantization
    # force to use only this type of quantization
    activations_quantization_params = advanced_parameters.activations_quantization_params
    if activations_quantization_params is None:
        activations_quantization_params = QuantizationParameters()

    activations_quantization_params.mode = QuantizationScheme.ASYMMETRIC
    advanced_parameters.activations_quantization_params = activations_quantization_params

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )
    nncf_graph = NNCFGraphFactory.create(copied_model)
    quantized_model = quantization_algorithm.apply(copied_model, nncf_graph, dataset=calibration_dataset)
    merge_conv_and_bias(quantized_model)

    # Magic. Without this call compiled model
    # is not preformant
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    quantized_model = _fold_conv_bn_qat(quantized_model)
    pm = PassManager([DuplicateDQPass()])

    quantized_model = pm(quantized_model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    quantized_model = pm(quantized_model).graph_module

    quantized_model.meta.update(original_graph_meta)
    quantized_model = _disallow_eval_train(quantized_model)

    return quantized_model
