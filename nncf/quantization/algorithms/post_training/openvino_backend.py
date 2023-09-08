# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import openvino.runtime as ov
from openvino.runtime import opset9 as opset
from tqdm import tqdm

from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.data.dataset import DataItem
from nncf.openvino.graph.metatypes.openvino_metatypes import OVIfMetatype
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVUpdateIfSubgraphCommand
from nncf.quantization.algorithms.post_training.backend import PostTrainingBackend


def _add_results(model: ov.Model, node: ov.Node) -> ov.Model:
    extra_model_outputs = []
    result_names = []
    for input in node.inputs():
        output = input.get_source_output()
        output_name = output.get_node().get_friendly_name()
        result_name = f"{output_name}/if_output"

        result = opset.result(output, name=result_name)

        tensor = result.get_output_tensor(0)
        current_names = tensor.get_names()
        current_names.add(result_name)
        tensor.set_names(current_names)
        result_names.append(result_name)
        extra_model_outputs.append(result)
    return ov.Model(
        results=extra_model_outputs,
        sinks=[op for op in model.get_ops() if op.get_type_name() == "Assign"],
        parameters=model.get_parameters(),
        name=model.friendly_name,
    )


class OVPostTrainingBackend(PostTrainingBackend):
    IF_OP_MODEL_INPUT_PORTS = (0, 1)

    @property
    def if_node_metatype(self):
        return OVIfMetatype

    @staticmethod
    def get_child_models(model: ov.Model, if_node: NNCFNode) -> List[Tuple[ov.Model, str]]:
        child_models = []
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        for port_id in OVPostTrainingBackend.IF_OP_MODEL_INPUT_PORTS:
            input_indices = [desc.input_index for desc in ov_node.get_input_descriptions(port_id)]
            input_names = [ov_node.input_values()[index].any_name for index in input_indices]
            child_models.append((ov_node.get_function(port_id), input_names))
        return child_models

    @staticmethod
    def get_if_input_name(model: ov.Model, if_node: NNCFNode) -> str:
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        return ov_node.input_values()[0].any_name

    @staticmethod
    def add_additional_outputs(model: ov.Model, if_node: NNCFNode) -> Tuple[ov.Model, List[str]]:
        name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        ov_node = name_to_node_mapping[if_node.node_name]
        return _add_results(model, ov_node)

    @staticmethod
    def create_update_subgraph_command(target_point, subgraph_model):
        return OVUpdateIfSubgraphCommand(target_point, subgraph_model)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def dump_model(model: ov.Model, dir: str, if_op: NNCFNode, if_op_model_input_port_id: int) -> None:
        name = if_op.node_name.replace("/", "")
        if if_op_model_input_port_id == 0:
            postfix = "then"
        if if_op_model_input_port_id == 1:
            postfix = "else"
        model_name = f"{name}_{postfix}.xml"
        model_path = Path(dir) / model_name
        ov.serialize(model, model_path)
