"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from nncf.common.quantization.structs import QuantizationMode


class WeightsStatisticFunction(Enum):
    MAX = 'max'
    MIN = 'min'
    ABS_MAX = 'abs_max'
    QUANTILE = 'quantile'
    ABS_QUANTILE = 'abs_quantile'


class ActivationsStatisticFunction(Enum):
    MAX = 'max'
    MIN = 'min'
    ABS_MAX = 'abs_max'
    QUANTILE = 'quantile'
    ABS_QUANTILE = 'abs_quantile'
    MEAN = 'mean'


class AggregationFunction(Enum):
    MEAN = 'mean'
    MAX = 'max'
    MIN = 'min'
    MEDIAN = 'median'
    MEAN_NO_OUTLIERS = 'mean_no_outliers'
    MEDIAN_NO_OUTLIERS = 'median_no_outliers'


class OverflowFix(Enum):
    ENABLE = 'enable'
    DISABLE = 'disable'
    FIRST_LAYER_ONLY = 'first_layer_only'


@dataclass
class WeightsEstimatorParameters():
    statistic_function:Optional[WeightsStatisticFunction] = None 
    statistic_function_params: Optional[Dict] = None
    clipping_value: Optional[float] = None


@dataclass
class ActivationsEstimatorParameters():
    statistic_function: Optional[ActivationsStatisticFunction] = None 
    statistic_function_params: Optional[Dict] = None
    aggregation_function: Optional[AggregationFunction] = None
    clipping_value: Optional[float] = None


@dataclass
class ActivationsQuantizationParameters:
    num_bits: Optional[int] = None
    mode: Optional[QuantizationMode] = None
    per_channel: Optional[bool] = None
    range_minimum_estimator: Optional[ActivationsEstimatorParameters] = None
    range_maximum_estimator: Optional[ActivationsEstimatorParameters] = None


@dataclass
class WeightsQuantizationParameters:
    num_bits: Optional[int] = None
    mode: Optional[QuantizationMode] = None
    per_channel: Optional[bool] = None 
    half_range: Optional[bool] = None
    narrow_range: Optional[bool] = None
    range_minimum_estimator: Optional[WeightsEstimatorParameters] = None
    range_maximum_estimator: Optional[WeightsEstimatorParameters] = None


@dataclass
class AdvancedQuantizationParameters:
    # quantization configurations for activations and weights
    activations: Optional[ActivationsQuantizationParameters] = None
    weights: Optional[WeightsQuantizationParameters] = None
       
    # BiasCorrection algorithm parameters
    apply_bias_correction_for_all_nodes: Optional[bool] = None
    bias_correction_threshold: Optional[float] = None
    
    # general parameters
    overflow_fix: Optional[OverflowFix] = None
    inplace_statistics: Optional[bool] = None
    
    # backend specific parameters
    backend_parameters: Optional[Dict[str, Any]] = None